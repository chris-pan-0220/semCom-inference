import os, time, pickle, sys, argparse, yaml
import torch
sys.path.append('./')
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
from ctypes import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import socket
import json
import pandas as pd

_snr = 10
_iscomplex = False # True 
channel_dim = 2

# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use CPU or GPU

class Decoder_Meta():
    
    # Note, `self` methods/attributes should be defined in child classes.
    def forward_rl(self, input_features, sample_max=None, multiple_sample=5, x_mask=None):

        if x_mask is not None:  # i.e., backbone is Transformer
            max_seq_len = input_features.shape[-1] // self.channel_dim
            input_features = smaple_n_times(multiple_sample, input_features.view(input_features.shape[0], max_seq_len, -1))
            input_features = self.from_channel_emb(input_features)
            x_mask = smaple_n_times(multiple_sample, x_mask)
        else: # LSTM
            def smaple_n_times(n, x):
                    if n>1:
                        x = x.unsqueeze(1) # Bx1x...
                        x = x.expand(-1, n, *([-1]*len(x.shape[2:])))
                        x = x.reshape(x.shape[0]*n, *x.shape[2:])
                    return x
            max_seq_len = 21  # we set the max sentence length to 20, plus an EOS token. You can adjust this value.
            input_features = smaple_n_times(multiple_sample, input_features)

        batch_size = input_features.size(0)
        state = self.init_hidden(input_features)

        seq = input_features.new_zeros((batch_size, max_seq_len), dtype=torch.long)
        seq_logprobs = input_features.new_zeros((batch_size, max_seq_len))
        seq_masks = input_features.new_zeros((batch_size, max_seq_len))
        it = input_features.new_zeros(batch_size, dtype=torch.long).fill_(self.sos_id)
        unfinished = it == self.sos_id
        for t in range(max_seq_len):
            logprobs, state = self._forward_step(it, state, input_features, x_mask)  # bs*vocab_size
            if sample_max:
                sample_logprobs, it = torch.max(logprobs.detach(), 1)
            else:
                it = torch.distributions.Categorical(logits=logprobs.detach()).sample()
                sample_logprobs = logprobs.gather(1, it.unsqueeze(1))  # gather the logprobs at sampled positions
            it = it.view(-1).long()
            sample_logprobs = sample_logprobs.view(-1)

            seq_masks[:, t] = unfinished
            it = it * unfinished.type_as(it)  # bs
            seq[:, t] = it
            seq_logprobs[:, t] = sample_logprobs

            unfinished = unfinished * (it != self.eos_id)  # update if finished according to EOS
            if unfinished.sum() == 0:
                break

        return seq, seq_logprobs, seq_masks
    def sample_max_batch(self, input_features, x_mask, decoding_constraint=1):
        self.eval()

        if x_mask:
            max_seq_len = input_features.shape[-1] // self.channel_dim
            input_features = self.from_channel_emb(input_features.view(input_features.shape[0], max_seq_len, -1))
        else:
            max_seq_len = 21

        batch_size = input_features.size(0)
        seq = input_features.new_zeros((batch_size, max_seq_len), dtype=torch.long)
        state = self.init_hidden(input_features)
        last_word_id = torch.zeros(batch_size, dtype=torch.long)
        it = input_features.new_zeros(batch_size, dtype=torch.long).fill_(self.sos_id)
        unfinished = it == self.sos_id
        for t in range(max_seq_len):

            logprobs, state = self._forward_step(it, state, input_features, x_mask)

            logprobs[:,self.pad_id] += float('-inf')  # do not generate <PAD>, <SOS> and <UNK>
            logprobs[:,self.sos_id] += float('-inf')
            logprobs[:,self.unk_id] += float('-inf')
            if decoding_constraint:  # do not generate last step word
                for idxx, xxx in enumerate(last_word_id):
                    logprobs[idxx, xxx] += float('-inf')
            it = torch.max(logprobs,1)[1]
            it = it * unfinished.type_as(it)  # once eos, output zero.
            seq[:,t] = it
            last_word_id = it.clone()
            unfinished = unfinished * (it != self.eos_id)

        return seq

class Embeds(nn.Module):
    def __init__(self, vocab_size, num_hidden):
        super(Embeds, self).__init__()
        self.emb = nn.Embedding(vocab_size, num_hidden, padding_idx=0)  # learnable params, nn.Embedding是用來將一個數字變成一個指定維度的向量的
        #vocab.GloVe(name='6B', dim=50, cache='../Glove') This is optional.

    def __call__(self, inputs):
        return self.emb(inputs)

class LSTMDecoder(nn.Module, Decoder_Meta):
    def __init__(self, channel_dim, embedds, vocab_size):
        super(LSTMDecoder, self).__init__()
        self.num_hidden = 512
        self.vocab_size = vocab_size
        self.channel_dim = channel_dim
        self.pad_id, self.sos_id, self.eos_id, self.unk_id = 0, 1, 2, 3
        self.word_embed_decoder = embedds
        self.from_channel_emb = nn.Linear(channel_dim, 2*self.num_hidden)
        self.lstmcell_decoder = nn.LSTMCell(input_size=self.num_hidden, hidden_size=self.num_hidden)
        self.linear_and_dropout_classifier_decoder = nn.Sequential(
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.Dropout(0.5), nn.Linear(self.num_hidden, self.vocab_size))
        
    def _forward_step(self, it, state, placeholder1, placeholder2):  # compatibility with Transformer backbone
        word_embs = self.word_embed_decoder(it)  # bs*word_emb
        _h, _c = self.lstmcell_decoder(word_embs, state)
        output = self.linear_and_dropout_classifier_decoder(_h)  # [bs, vocab_size]
        logprobs = F.log_softmax(output, dim=-1)  # [bs*vocab_size] In LSTM cell, we always run one step, for T times.
        return logprobs, (_h, _c)

    def forward_ce(self, input_features, gt_captions, src_mask=None, ss_prob=None):
        assert ss_prob is not None, 'must provide ss_prob'
        batch_size = gt_captions.size(0)
        state = self.init_hidden(input_features)
        outputs = []
        for i in range(gt_captions.size(1)):  # length of captions.
            if self.training and i >= 1 and ss_prob > 0.0:  # otherwise no need to sample
                sample_prob = input_features.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < ss_prob
                it = gt_captions[:, i - 1].clone()  # teacher forcing
                if sample_mask.sum() != 0:
                    sample_ind = sample_mask.nonzero().view(-1)
                    prob_prev = outputs[i - 1].detach().exp()  # bs*vocab_size, fetch prev distribution
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                    it = input_features.new_zeros(batch_size, dtype=torch.long).fill_(self.sos_id) if i==0 else \
                        gt_captions[:, i-1].clone()  # it is the input of decoder.

            logprobs, state = self._forward_step(it, state, None, None)  # placeholders, for compatibility
            outputs.append(logprobs)

        outputs = torch.stack(outputs, dim=1)  # [bs, max_len, vocab_size]
        return outputs
    
    def init_hidden(self, input):
        x = self.from_channel_emb(input)
        return (x[:,:self.num_hidden],  x[:,self.num_hidden:]) # split into half


    def forward(self, x, placeholder):
        # x: [bs, T]
        src_mask = (x != self.pad_id).unsqueeze(1)
        x = self.word_embed_encoder(x)
        x = self.PE(x)  # positional embedding
        x = self.encoder(x, src_mask)  # [bs, T, d_model]
        embedding_channel = self.to_chanenl_embedding(x)
        embedding_channel = embedding_channel.view(embedding_channel.shape[0],-1)
        return embedding_channel, src_mask

def convert_binary_to_float_v1(binary_tensor):
    # Tensor to hold the floating point values
    # restored_tensor = torch.zeros(binary_tensor.shape[0], binary_tensor.shape[1], dtype=torch.float32)
    restored_values = torch.zeros(binary_tensor.shape[1], dtype=torch.float32).to(device)
    # Dictionary to map the binary representation to its float value
    fractional_mapping_v1 = {

(1, 1): 1.625,
(0, 1): 1.625,
(1, 0): 1.625,

(0, 0): 0.375,
    }
    fractional_mapping_v2= {

(1, 1): 0.125,
(1, 0): 0.125,
(0, 1): 0.125,

(0, 0): 1.875,

    }

    for i, mapping in enumerate([fractional_mapping_v1, fractional_mapping_v2]):
        # Accessing the entire 4-bit binary number for each dimension
        binary_bits = tuple(binary_tensor[0][i])  # Accessing the row for 4-bit binary

        # Convert each tensor element in the tuple to an integer
        binary_bits_int = tuple(bit.item() for bit in binary_bits)

        # Use the appropriate mapping for each "dimension"
        restored_values[i] = mapping[binary_bits_int]

    return restored_values


with open('./config/test.yaml', 'r') as file:
    config = yaml.safe_load(file)

decoder_path = config['model']['decoder_path']
embeds_shared_path = config['model']['embeds_shared_path']
train_dict_path = config['train']['train_dict_path']
socket_path = config['socket']['decoder_path']

print(f"decoder path: {decoder_path}")
print(f"embeds path: {embeds_shared_path}")
print(f"train dict path: {train_dict_path}")
print(f"socket path: {socket_path}")

embeds_shared = Embeds(vocab_size=24, num_hidden=512).to(device)
decoder = LSTMDecoder(channel_dim=channel_dim, embedds=embeds_shared, vocab_size = 24).to(device)

decoder = decoder.eval()
embeds_shared = embeds_shared.eval()

max_test_time = 10000
test_time = 0

recv_time = None
inference_time = None
encode_time = None
decode_time = None
bleu_time = None
send_time = None
total_time = None

def decoding(input_data: torch.tensor,
            decoder: nn.Module):

    total_errors_accumulated = 0
    total_bits_accumulated = 0

    with torch.no_grad():

        """Dequantized""" 
        pred8 = convert_binary_to_float_v1(input_data)
        # print("bit vector convert float_point :")
        # m = 0 # Given m is 1
        # n = 0 # Given m is 1   
        # X =  0.078125
        # Y =  0.109375
        # pred8[0, 0] -= m * X  # Subtract only from the first element
        # pred8[0, 1] += n * Y  # Subtract only from the first element
        pred8 = pred8.unsqueeze(0)
        # print(pred8)
        pred8 = pred8 - 0.9999
        # print("Dequantized :")
        # print(pred8)
     
        """decode"""
        st = time.time()
        output = decoder.sample_max_batch(pred8, None)
        end = time.time()
        # print(f'decoder time: {(end-st)*1000:.3f} ms')
        decode_time[test_time] = end - st
        # print("decoder :")
        # print(output)

        calculated_BER = total_errors_accumulated / total_bits_accumulated if total_bits_accumulated > 0 else 0

        # print(f"Total number of bits: {total_bits_accumulated}")
        # print(f"Total number of errors: {total_errors_accumulated}")
        # print(f"Calculated BER: {calculated_BER}")
    return output

def warmup(input_data:torch.tensor, decoder: nn.Module):
    print('warm up 5 times')
    for _ in range(5):
        decoding(input_data, decoder)

def stat_table(name_list, stat_time_list):
    print('perf in ms: ')
    combined_df = pd.concat([(pd.Series(x)*1000).describe() for x in stat_time_list], axis=1)
    combined_df.columns = name_list
    return combined_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    args = parser.parse_args()

    max_test_time = args.n
    print('test time: ', max_test_time)
    recv_time = [0 for _ in range(max_test_time)]
    inference_time = [0 for _ in range(max_test_time)]
    encode_time = [0 for _ in range(max_test_time)]
    decode_time = [0 for _ in range(max_test_time)]
    bleu_time = [0 for _ in range(max_test_time)]
    send_time = [0 for _ in range(max_test_time)]
    total_time = [0 for _ in range(max_test_time)]

    # dict_train = pickle.load(open('./1.train_dict.pkl', 'rb'))
    dict_train = pickle.load(open(train_dict_path, 'rb'))
    rev_dict = {vv: kk for kk, vv in dict_train.items()}

    success_count = 0
    failure_count = 0

    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    embeds_shared.load_state_dict(torch.load(embeds_shared_path,  map_location=device))
    
    warmup(input_data=torch.tensor([[[0,1], [1,0]]]).to(device), decoder=decoder)

    #  Set the path for the Unix socket
    # socket_path = 'semcom_decoder'

    # remove the socket file if it already exists
    try:
        os.unlink(socket_path)
    except OSError:
        if os.path.exists(socket_path):
            raise

    # Create the Unix socket server
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    # Bind the socket to the path
    server.bind(socket_path)

    # Listen for incoming connections
    server.listen(1)

    # accept connections
    print('Server is listening for incoming connections...')
    connection, client_address = server.accept()

    try:
        print('Connection: ', str(connection))
        # receive data from the client

        while True:
            input_str = connection.recv(256) # receive data from ...
            time_t1 = time.time()

            if not input_str:
                break

            input_str = input_str.decode()
            # print('input str: ', input_str)            
            input_data = torch.tensor(json.loads(input_str)).to(device=device) # check `input_data` type
            # print('input data: ', input_data)
            
            output = decoding(input_data, decoder)
            output = output.cpu().numpy()[0]
            res = ' '.join(rev_dict[x] for x in output if x!=0 and x!=2)  # remove 'PAD' and 'EOS'
            time_t2 = time.time()
            # print('--------------Candidate sentence---------------')
            # print('{}'.format(res))
        
            # print('-----------------------------------------------')
            # print('------------------Comparison-------------------')
            # print('-----------------------------------------------')

            sent_a_reference = 'planets engage in an eternal graceful dance around sun rise'.split()
            # print('sent_a_reference_sentence = {} '.format(sent_a_reference))
        
            sent_b_reference = 'moon casts gentle soft glow across dark darkness took blue'.split()
            # print('sent_b_reference_sentence = {} '.format(sent_b_reference))

            sent_a_candidate = ' {} '.format(res).split()
            # print('sent_a_candidate_sentence = {} '.format(sent_a_candidate))

            # print('-----------------------------------------------')
            # print('-----------------BLEU-4 score------------------')
            # print('-----------------------------------------------')
            """BLEU score"""
            smoothie = SmoothingFunction().method2
            bleu1= bleu([sent_a_reference], sent_a_candidate, smoothing_function=smoothie)
            smoothie = SmoothingFunction().method2
            bleu2= bleu([sent_b_reference], sent_a_candidate, smoothing_function=smoothie)
            time_t3 = time.time()
            # print('bleu score 1 (sent_a_reference_sentence, sent_a_candidate_sentence)= {} '.format(bleu1))
            
            if (bleu1 > bleu2):
                # print('bleu score 1 > bleu score 2 = {} '.format(bleu1))
                # print('sent_a_reference_sentence = {} '.format(sent_a_reference))
                # print('sent_a_candidate_sentence = {} '.format(sent_a_candidate))
                # print('Confirmation sent_a_reference_sentence sent message \n')
                # pass
                connection.sendall('planets engage in an eternal graceful dance around sun rise'.encode())
                success_count += 1
                time_t4 = time.time()
                
            else:
                # print('bleu score 2 > bleu score 1 = {} '.format(bleu2))
                
                connection.sendall('moon casts gentle soft glow across dark darkness took blue'.encode())
                failure_count += 1
                time_t4 = time.time()

            inference_time[test_time] = time_t2 - time_t1
            bleu_time[test_time] = time_t3 - time_t2
            send_time[test_time] = time_t4 - time_t3
            total_time[test_time] = time_t4 - time_t1

            # Print the success and failure counts
            # print('Success count:', success_count)
            # print('Failure count:', failure_count)
            test_time += 1
    finally:
        # Print the success and failure counts
        print('Success count:', success_count)
        print('Failure count:', failure_count)

        # stat('recv', recv_time)
        # stat('decode', decode_time)
        # stat('inference', inference_time)
        # stat('bleu', bleu_time)
        # stat('send', send_time)
        # stat('total', total_time)
        name_list = ['decode', 'inference', 'bleu', 'send', 'total']
        stat_time_list = [decode_time, inference_time, bleu_time, send_time, total_time]
        print(stat_table(name_list, stat_time_list))

        # close the connection
        connection.close()
        # remove the socket file
        os.unlink(socket_path)

