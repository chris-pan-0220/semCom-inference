#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

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

    struct sockaddr_un decoder_addr;
    int decoder_fd;
    char buffer[BUFFER_SIZE], recv_buffer[BUFFER_SIZE], encoder_buffer[BUFFER_SIZE];

    // create decoder socket
    if ((decoder_fd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
        perror("socket error");
        exit(EXIT_FAILURE);
    }

    // set socket addr
    memset(&decoder_addr, 0, sizeof(struct sockaddr_un));
    decoder_addr.sun_family = AF_UNIX;
    strncpy(decoder_addr.sun_path, DECODER_PATH, sizeof(decoder_addr.sun_path) - 1);

    // connect to decoder
    if (connect(decoder_fd, (struct sockaddr *)&decoder_addr, sizeof(decoder_addr)) == -1) {
        perror("Error:");
        exit(1);
    }

    struct timeval time_start;
    struct timeval time_end;

    int exceed_10ms = 0;
    double diff;
    
    for(int t = 0;t < TEST_TIME;t++){
        gettimeofday(&time_start, NULL); 
        write(decoder_fd, "[[[0, 0], [1, 0]]]", strlen("[[[0, 0], [1, 0]]]"));
        read(decoder_fd, buffer, BUFFER_SIZE);    
        gettimeofday(&time_end, NULL); 
        printf("Client: Recieve response\n\t`%s`\nfrom decoder\n", buffer);
        diff = ((double)((time_end.tv_sec - time_start.tv_sec)*1000000L + time_end.tv_usec - time_start.tv_usec)) / 1000;
        printf("Inference time: %.3f ms\n", diff);
        if(!(diff < 10))
            exceed_10ms += 1;
    }

    printf("%d/%d exceed 10ms\n", exceed_10ms, TEST_TIME);

    close(decoder_fd);
    return 0;
}
