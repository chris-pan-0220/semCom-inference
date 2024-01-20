#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#define SOCKET_PATH "semcom_recver"
#define DECODER_PATH "semcom_decoder"
#define BUFFER_SIZE 256

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s -n TEST_TIME\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "-n") != 0) {
        fprintf(stderr, "Error: missing argument `-n`\n");
        return 1;
    }

    const int TEST_TIME = atoi(argv[2]);

    if (TEST_TIME < 1) {
        fprintf(stderr, "Error:  argument `TEST_TIME` must be greater than 0\n");
        return 1;
    }

    struct sockaddr_un recver_addr, decoder_addr;
    int recver_fd, sender_fd, decoder_fd;
    char buffer[BUFFER_SIZE], recv_buffer[BUFFER_SIZE], encoder_buffer[BUFFER_SIZE];

    // create recver socket
    if ((recver_fd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
        perror("socket error");
        exit(EXIT_FAILURE);
    }

    // create decoder socket
    if ((decoder_fd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
        perror("socket error");
        exit(EXIT_FAILURE);
    }

    // set socket addr
    memset(&recver_addr, 0, sizeof(struct sockaddr_un));
    recver_addr.sun_family = AF_UNIX;
    strncpy(recver_addr.sun_path, SOCKET_PATH, sizeof(recver_addr.sun_path) - 1);

    // set socket addr
    memset(&decoder_addr, 0, sizeof(struct sockaddr_un));
    decoder_addr.sun_family = AF_UNIX;
    strncpy(decoder_addr.sun_path, DECODER_PATH, sizeof(decoder_addr.sun_path) - 1);

    // connect to decoder
    if (connect(decoder_fd, (struct sockaddr *)&decoder_addr, sizeof(decoder_addr)) == -1) {
        perror("Error:");
        exit(1);
    }

    // bind recver socket
    unlink(SOCKET_PATH); // ???
    if (bind(recver_fd, (struct sockaddr*)&recver_addr, sizeof(struct sockaddr_un)) == -1) {
        perror("bind error");
        exit(EXIT_FAILURE);
    }

    if (listen(recver_fd, 1) == -1) {
        perror("listen error");
        exit(EXIT_FAILURE);
    }

    struct timeval time_start;
    struct timeval time_end;

    int exceed_10ms = 0;
    double diff;

    printf("Waiting for connections...\n");

    int done = 0;
    while (1) {
        if(done == 1)
            break;
        if ((sender_fd = accept(recver_fd, NULL, NULL)) == -1) {
            perror("accept fail");
            continue;
        }

        printf("Client connected\n");

        // read data from sender
        while (1) {
            ssize_t read_bytes = read(sender_fd, recv_buffer, BUFFER_SIZE - 1);
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

                // without response
                diff = ((double)((time_end.tv_sec - time_start.tv_sec)*1000000L + time_end.tv_usec - time_start.tv_usec)) / 1000;
                printf("Encoding time: %.3f ms\n", diff);
                if(!(diff < 10))
                    exceed_10ms += 1;
                
                write(sender_fd, "OK\0", strlen("OK\0"));
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

    printf("%d/%d exceed 10ms\n", exceed_10ms, TEST_TIME);

    close(recver_fd);
    return 0;
}
