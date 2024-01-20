#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>

#define EXAMPLE_LEN 1
#define MAX_PACKET_LEN 256

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

    int encoder_socket, recver_socket;
    struct sockaddr_un encoder_addr, recver_addr;
    int connection_result;

    const char * const SemanticRL_example[EXAMPLE_LEN] = {
        "planets engage in an eternal graceful dance around sun rise\0"
        // "planets engag in an eternal greceful dance arund sun rise\0"
    };
    // printf("example: %s\n", SemanticRL_example[0]);

    char *response = (char*)malloc(sizeof(char) * MAX_PACKET_LEN);

    /*connect to encoder*/
    encoder_socket = socket(AF_UNIX, SOCK_STREAM, 0);

    encoder_addr.sun_family = AF_UNIX;
    strcpy(encoder_addr.sun_path, "semcom_encoder");// unix socket path

    connection_result = connect(encoder_socket, (struct sockaddr *)&encoder_addr, sizeof(encoder_addr));

    if (connection_result == -1) {
        perror("Error:");
        exit(1);
    }

    /*connect to recver*/
    recver_socket = socket(AF_UNIX, SOCK_STREAM, 0);

    recver_addr.sun_family = AF_UNIX;
    strcpy(recver_addr.sun_path, "semcom_recver");// unix socket path

    connection_result = connect(recver_socket, (struct sockaddr *)&recver_addr, sizeof(recver_addr));

    if (connection_result == -1) {
        perror("Error:");
        exit(1);
    }

    struct timeval time_start;
    struct timeval time_end;

    int exceed_10ms = 0;
    double diff;
    
    for(int t = 0;t < TEST_TIME;t++){
        for(int i = 0;i < EXAMPLE_LEN;i++){
            gettimeofday(&time_start, NULL); 
            // send input to encoder
            write(encoder_socket, SemanticRL_example[i], strlen(SemanticRL_example[i]));
            // read response from encoder
            read(encoder_socket, response, MAX_PACKET_LEN);    
            gettimeofday(&time_end, NULL); // encoding time
            printf("Client: Recieve response `%s` from encoder\n", response);
            
            // printf("Client: Recieve response\n\t`%s`\nfrom encoder\n", response);
            
            diff = ((double)((time_end.tv_sec - time_start.tv_sec)*1000000L + time_end.tv_usec - time_start.tv_usec)) / 1000;
            printf("Encoding time: %.3f ms\n", diff);
            if(!(diff < 10))
                exceed_10ms += 1;

            // send encoding response to recver
            write(recver_socket, response, strlen(response));
            // wait for recver ok
            read(recver_socket, response, MAX_PACKET_LEN);
            // write(recver_socket, SemanticRL_example[i], strlen(SemanticRL_example[i]));
        }
    }

    printf("%d/%d exceed 10ms\n", exceed_10ms, TEST_TIME);
    
    close(encoder_socket);
    close(recver_socket);
    free(response);
}