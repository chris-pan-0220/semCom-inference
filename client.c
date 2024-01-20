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

    int server_socket;
    struct sockaddr_un server_addr;
    int connection_result;

    const char * const SemanticRL_example[EXAMPLE_LEN] = {
        "planets engage in an eternal graceful dance around sun rise\0"
        // "planets engag in an eternal greceful dance arund sun rise\0"
    };
    // printf("example: %s\n", SemanticRL_example[0]);

    char *response = (char*)malloc(sizeof(char) * MAX_PACKET_LEN);

    server_socket = socket(AF_UNIX, SOCK_STREAM, 0);

    server_addr.sun_family = AF_UNIX;
    strcpy(server_addr.sun_path, "semCom_benchmark");// unix socket path

    connection_result = connect(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr));

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
            write(server_socket, SemanticRL_example[i], strlen(SemanticRL_example[i]));
            read(server_socket, response, MAX_PACKET_LEN);    
            gettimeofday(&time_end, NULL); 
            // printf("Client: Recieve response\n\t`%s`\nfrom server\n", response);
            diff = ((double)((time_end.tv_sec - time_start.tv_sec)*1000000L + time_end.tv_usec - time_start.tv_usec)) / 1000;
            // printf("Inference time: %.3f ms\n", diff);
            if(!(diff < 10))
                exceed_10ms += 1;
        }
    }

    printf("%d/%d exceed 10ms\n", exceed_10ms, TEST_TIME);
    
    close(server_socket);
    free(response);
}