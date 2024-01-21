#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>
#include <arpa/inet.h>

#define SOCKET_PATH "semcom_recver"
#define DECODER_PATH "semcom_decoder"
#define BUFFER_SIZE 256

int main(int argc, char* argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s -n TEST_TIME -p PORT\n", argv[0]);
        return 1;
    }

    char *port = NULL;
    int number = 0;
    int portProvided = 0, numberProvided = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            number = atoi(argv[i + 1]);
            numberProvided = 1;
            i++;
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            port = argv[i + 1];
            portProvided = 1;
            i++; 
        }
    }

    if (!portProvided || !numberProvided) {
        fprintf(stderr, "Error: Missing required arguments\n");
        return 1;
    }

    int recver_port = atoi(port);
    printf("recver port: %d\n", recver_port);

    const int TEST_TIME = number;

    struct sockaddr_un decoder_addr;
    struct sockaddr_in recver_addr;
    int recver_fd, sender_fd, decoder_fd;
    char buffer[BUFFER_SIZE], recv_buffer[BUFFER_SIZE], encoder_buffer[BUFFER_SIZE];

    /************connect to decoder**************/

    // create decoder socket
    if ((decoder_fd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
        perror("socket error");
        exit(EXIT_FAILURE);
    }

    // set decoder socket addr
    memset(&decoder_addr, 0, sizeof(struct sockaddr_un));
    decoder_addr.sun_family = AF_UNIX;
    strncpy(decoder_addr.sun_path, DECODER_PATH, sizeof(decoder_addr.sun_path) - 1);

    // connect to decoder
    if (connect(decoder_fd, (struct sockaddr *)&decoder_addr, sizeof(decoder_addr)) == -1) {
        perror("Error:");
        exit(1);
    }

    /************create recver socket server**************/
    
    // create recver socket
    if ((recver_fd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        perror("socket error");
        exit(EXIT_FAILURE);
    }

    // set recver socket addr
    memset(&recver_addr, 0, sizeof(struct sockaddr_in));
    recver_addr.sin_family = AF_INET;
    recver_addr.sin_addr.s_addr = INADDR_ANY;
    recver_addr.sin_port = htons(recver_port);

    // bind recver socket 
    if (bind(recver_fd, (struct sockaddr *)&recver_addr, sizeof(recver_addr)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(recver_fd, 1) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    
    struct timeval time_start;
    struct timeval time_end;
    
    int done = 0;
    int exceed_10ms = 0;
    double diff;

    printf("Waiting for connections...\n");

    while(1){
        if(done)
            break;
        if ((sender_fd = accept(recver_fd, (struct sockaddr*)NULL, NULL)) < 0) {
            perror("accept");
            continue;
        }

        printf("Client connected\n");

        while (1) {
            ssize_t read_bytes = recv(sender_fd, recv_buffer, BUFFER_SIZE - 1, 0);
            if (read_bytes > 0) {
                recv_buffer[read_bytes] = '\0'; // ensure string is `\0` terminated
                // printf("Received data from sender: %s\n", buffer);
                printf("Received data `%s` from sender\n", recv_buffer);
                
                // send data to decoder
                gettimeofday(&time_start, NULL); 
                write(decoder_fd, recv_buffer, strlen(recv_buffer));
                // read response from decoder
                read_bytes = read(decoder_fd, encoder_buffer, BUFFER_SIZE - 1);
                if (read_bytes < 0){ // client close connection or error occured
                    perror("read from encoder error");
                    break;
                }
                encoder_buffer[read_bytes] = '\0';
                gettimeofday(&time_end, NULL);

                printf("Received response `%s` from decoder\n", buffer);

                diff = ((double)((time_end.tv_sec - time_start.tv_sec)*1000000L + time_end.tv_usec - time_start.tv_usec)) / 1000;
                printf("Encoding time: %.3f ms\n", diff);
                if(!(diff < 10))
                    exceed_10ms += 1;
                
                send(sender_fd, "OK\0", strlen("OK\0"), 0);
            } else {
                if (read_bytes < 0) // client close connection or error occured
                    perror("read from sender error");
                break;
            }
        }

        printf("Client disconnected\n");
        done = 1;
        close(sender_fd);
    }

    // int done = 0;
    // while (1) {
    //     if(done == 1)
    //         break;
    //     if ((sender_fd = accept(recver_fd, NULL, NULL)) == -1) {
    //         perror("accept fail");
    //         continue;
    //     }

    //     printf("Client connected\n");

    //     // read data from sender
    //     while (1) {
    //         ssize_t read_bytes = read(sender_fd, recv_buffer, BUFFER_SIZE - 1);
    //         if (read_bytes > 0) {
    //             recv_buffer[read_bytes] = '\0'; // ensure string is `\0` terminated
    //             // printf("Received data from sender: %s\n", buffer);
    //             printf("Received data `%s` from sender\n", recv_buffer);
                
    //             // send data to decoder
    //             gettimeofday(&time_start, NULL); 
    //             write(decoder_fd, recv_buffer, strlen(recv_buffer));
    //             // read response from decoder
    //             read_bytes = read(decoder_fd, encoder_buffer, BUFFER_SIZE - 1);
    //             if (read_bytes < 0){ // client close connection or error occured
    //                 perror("read from encoder error");
    //                 break;
    //             }
    //             encoder_buffer[read_bytes] = '\0';
    //             gettimeofday(&time_end, NULL);

    //             printf("Received response `%s` from decoder\n", buffer);

    //             // without response
    //             diff = ((double)((time_end.tv_sec - time_start.tv_sec)*1000000L + time_end.tv_usec - time_start.tv_usec)) / 1000;
    //             printf("Encoding time: %.3f ms\n", diff);
    //             if(!(diff < 10))
    //                 exceed_10ms += 1;
                
    //             write(sender_fd, "OK\0", strlen("OK\0"));
    //         } else {
    //             if (read_bytes < 0) // client close connection or error occured
    //                 perror("read from sender error");
    //             break;
    //         }
    //     }

    //     printf("Client disconnected\n");
    //     done = 1;
    //     close(sender_fd);
    // }

    printf("%d/%d exceed 10ms\n", exceed_10ms, TEST_TIME);

    close(recver_fd);
    return 0;
}
