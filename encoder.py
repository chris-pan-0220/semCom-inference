import os, time, pickle, sys, argparse, yaml
import torch
sys.path.append('./')
from ctypes import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import socket
import os
import json
import pandas as pd

_snr = 10
_iscomplex = False # True 
channel_dim = 2

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
    
class LSTMEncoder(nn.Module):
    def __init__(self, channel_dim, embedds):
        super(LSTMEncoder, self).__init__()

        self.num_hidden = 512
        self.pad_id = 0
        self.word_embed_encoder = embedds
        self.lstm_encoder = nn.LSTM(input_size=self.num_hidden, hidden_size=self.num_hidden,
                                    num_layers=2, bidirectional=True, batch_first=True)
        self.to_chanenl_embedding = nn.Sequential(nn.Linear(2*self.num_hidden, 2*self.num_hidden), nn.ReLU(),
                                                    nn.Linear(2*self.num_hidden, channel_dim))
        
    def forward(self, x, len_batch):
        # x: [batch, T]
        self.lstm_encoder.flatten_parameters()
        word_embs = self.word_embed_encoder(x)
       
        word_embs_packed = pack_padded_sequence(word_embs, len_batch, enforce_sorted=True, batch_first=True)
        
        output, state = self.lstm_encoder(word_embs_packed)  # output is a packed seq
        (_data ,_len) = pad_packed_sequence(output, batch_first=True)
        forward_ = word_embs.new_zeros(x.size(0), self.num_hidden, dtype=torch.float)
        backward_ = word_embs.new_zeros(x.size(0), self.num_hidden, dtype=torch.float)
       
        for i in range(x.size(0)):
            forward_[i,:] = _data[i, _len[i]-1, :self.num_hidden]  # we take the last forward step
            backward_[i,:] = _data[i, 0, self.num_hidden:]  # and the first backward step
        embedding_channel = self.to_chanenl_embedding(torch.cat([forward_, backward_], dim=1))
        return embedding_channel, None  # src_mask is None

class Normlize_tx:
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex
    def apply(self, _input):
        _dim = _input.shape[1]//2 if self._iscomplex else _input.shape[1]
        _norm = _dim**0.1 / torch.sqrt(torch.sum(_input**2, dim=1))
        return _input*_norm.view(-1,1)

class Channel:
    # returns the message when passed through a channel.
    # AGWN, Fading
    # Note that we need to make sure that the colle map will not change in this
    # step, thus we should not use *= and +=.
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex

    def ideal_channel(self, _input):
        return _input

    def awgn(self, _input, _snr):
        _std = (10**(-_snr/10.)/2)**0.5 if self._iscomplex else (10**(-_snr/10.))**0.5  # for complex signals.
        _input = _input + torch.randn_like(_input) * _std

        return _input
    
    def smaple_n_times(n, x):
        if n>1:
            x = x.unsqueeze(1) # Bx1x...
            x = x.expand(-1, n, *([-1]*len(x.shape[2:])))
            x = x.reshape(x.shape[0]*n, *x.shape[2:])
        return x

def quantize_and_convert_to_binary_with_integer_part(input_tensor, error_boundary):
    # Define quantization ranges and values
    quantization_ranges = [
                           (-2.0, -1.751), (-1.75, -1.51), (-1.5, -1.251), (-1.25, -1.001),
                           (-1.0, -0.751), (-0.75, -0.51), (-0.5, -0.251), (-0.25, -0.001),
                           (0.0, 0.249), (0.25, 0.499), (0.5, 0.749), (0.75, 0.99), 
                           (1.0, 1.249), (1.25, 1.499), (1.5, 1.749), (1.75, 1.99)]
    quantization_values = [-1.875, -1.625, -1.375, -1.125, -0.875, -0.625, -0.375, -0.125,
                           0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875]
    # Quantize the tensor
    quantized_tensor = input_tensor.clone()
    for i in range(input_tensor.shape[0]):
        for j in range(input_tensor.shape[1]):
            value = input_tensor[i][j] + 0.9999  # Normalize the input to the range 0 to 1.9998
        
            for q_range, q_value in zip(quantization_ranges, quantization_values):
                lower_bound, upper_bound = q_range
                if lower_bound <= value < upper_bound:
                    quantized_tensor[i][j] = q_value
                    break
    # print(quantized_tensor)
    # Convert to binary representation
    binary_tensor = torch.zeros(input_tensor.shape[0], input_tensor.shape[1], 5, dtype=torch.int32)
    for i in range(quantized_tensor.shape[0]):
        for j in range(quantized_tensor.shape[1]):
            value = quantized_tensor[i][j]

            # Set the integer part bits (first bit)
            integer_part = int(value)
            binary_tensor[i, j, :0] = 1  

            # Set the integer part bits (6th to 10th bits)
            # For -0.9252, set to 0; for others set to 1
            if (value == 0.125) or(value == 0.375) or (value == 0.625) or (value == 0.875):  # Quantized value for -0.9252
                binary_tensor[i, j, 0:3] = 0
            else:
                binary_tensor[i, j, 0:3] = 1

          

            # Set the integer part bits (6th to 10th bits) based on the integer part of value
            if (value == 1.125) or (value == 1.375) or (value == 1.625) or (value == 1.875):
                binary_tensor[i, j, 0:3] = 1  # Integer part 1
            else:
                binary_tensor[i, j, 0:3] = 0  # Integer part 0

            # Set the fractional part bits (second and third bits)
            fractional_part = value - integer_part
            if fractional_part < 0.25:
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 0
            elif fractional_part < 0.5:
                binary_tensor[i, j, 3] = 0
                binary_tensor[i, j, 4] = 1
            elif fractional_part < 0.75:
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 0
            else:
                binary_tensor[i, j, 3] = 1
                binary_tensor[i, j, 4] = 1

    return binary_tensor

with open('./config/test.yaml', 'r') as file:
    config = yaml.safe_load(file)

encoder_path = config['model']['encoder_path']
embeds_shared_path = config['model']['embeds_shared_path']
train_dict_path = config['train']['train_dict_path']

print(f"encoder path: {encoder_path}")
print(f"embeds path: {embeds_shared_path}")
print(f"train dict path: {train_dict_path}")

# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use CPU or GPU
embeds_shared = Embeds(vocab_size=24, num_hidden=512).to(device)
encoder = LSTMEncoder(channel_dim=channel_dim, embedds=embeds_shared).to(device)

encoder = encoder.eval()
embeds_shared = embeds_shared.eval()

normlize_layer = Normlize_tx(_iscomplex=_iscomplex)
channel = Channel(_iscomplex=_iscomplex)

max_test_time = 10000
test_time = 0

recv_time = None
inference_time = None
encode_time = None
decode_time = None
bleu_time = None
send_time = None
total_time = None

def encoding(input_data: torch.tensor,
            encoder: nn.Module,
            normlize_layer: nn.Module,
            len_batch: torch.tensor):

    total_errors_accumulated = 0
    total_bits_accumulated = 0

    with torch.no_grad():

        """encoding"""
        st = time.time()
        output, _ = encoder(input_data, len_batch)
        end = time.time()
        encode_time[test_time] = end - st
        # print(f'enoder time: {(end-st)*1000:.3f} ms')
        # print("encoder :")
        # print(output)

        """normalize"""
        output = normlize_layer.apply(output)
        # print("normlize_layer :")
        # print(output)
        
        """quantize_and_convert_to_binary_with_integer_part"""
        error_boundary = 0.125 
        pred1 = quantize_and_convert_to_binary_with_integer_part(output, error_boundary).to(device)
        # print("Quantized :")
        # print(f"Original Fixed Point:\n{pred1}")
        # print(pred1)
  
        """Channel"""
        replaced_tensor = torch.tensor([[[0, 0], [0, 0]]]).to(device)
        # print(replaced_tensor)

        # # Channel noise
        # print("Channel noise :")
        BER_target = 0.26354  # 0.002561 

        error_mask = (torch.rand(replaced_tensor.shape) < BER_target).to(device)
        received_pred1 = replaced_tensor ^ error_mask.long()
        num_errors = torch.sum(replaced_tensor != received_pred1)
        total_bits_accumulated += replaced_tensor.numel()
        total_errors_accumulated += num_errors.item()
        # print(f"Received Fixed Point with BER {BER_target}:\n{received_pred1}")
        # print(received_pred1)
        # quantization_values = [0.375, 0.458333, 0.541666, 0.624999, 0.708332, 0.791665, 0.874998, 0.958331, 1.041664, 1.124997, 1.20833, 1.291663, 1.374996, 1.458329, 1.541662, 1.625]
    return received_pred1

def warmup(encoder: nn.Module,
            normlize_layer: nn.Module):
    print('warmup 5 times')
    input_str = 'planets engage in an eternal graceful dance around sun rise'
    input_vector = [dict_train[x] for x in input_str.split(' ')] + [2]
    input_len = len(input_vector)
    input_vector = torch.tensor(input_vector).to(device)

    for _ in range(5):
        encoding(input_vector.unsqueeze(0),
                encoder,
                normlize_layer,
                len_batch=torch.tensor(input_len).view(-1, ))

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

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    embeds_shared.load_state_dict(torch.load(embeds_shared_path,  map_location=device))
    
    warmup(encoder=encoder, normlize_layer=normlize_layer)

    #  Set the path for the Unix socket
    socket_path = 'semcom_encoder'

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
            input_str = connection.recv(256)
            time_t1 = time.time()

            if not input_str:
                break

            input_str = input_str.decode()
            input_vector = [dict_train[x] for x in input_str.split(' ')] + [2]
            input_len = len(input_vector)
            input_vector = torch.tensor(input_vector).to(device)

            output = encoding(input_vector.unsqueeze(0),
                              encoder,
                              normlize_layer,
                              len_batch=torch.tensor(input_len).view(-1, ))
            # print('output:')
            # print(output)
            time_t2 = time.time()
            connection.sendall(json.dumps(output.tolist()).encode())
            time_t3 = time.time()
                
            # recv time:
            inference_time[test_time] = time_t2 - time_t1
            send_time[test_time] = time_t3 - time_t2
            total_time[test_time] = time_t3 - time_t1
                
            test_time += 1
    finally:

        # stat('recv', recv_time)
        # stat('encode', encode_time)
        # stat('inference', inference_time)
        # stat('send', send_time)
        # stat('total', total_time)

        name_list = ['encode', 'inference', 'send', 'total']
        stat_time_list = [encode_time, inference_time, send_time, total_time]
        
        print(stat_table(name_list, stat_time_list))
        
        
        # close the connection
        connection.close()
        # remove the socket file
        os.unlink(socket_path)

