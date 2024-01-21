#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <arpa/inet.h>

#define EXAMPLE_LEN 1
#define MAX_PACKET_LEN 256

int isValidIpAddress(const char *ip) {
    struct sockaddr_in sa;
    return inet_pton(AF_INET, ip, &(sa.sin_addr)) != 0;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s -n TEST_TIME -p RECVER_IP:PORT\n", argv[0]);
        return 1;
    }

    char *ip = NULL;
    int number = 0;
    int ipProvided = 0, numberProvided = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            number = atoi(argv[i + 1]);
            numberProvided = 1;
            i++;
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            ip = argv[i + 1];
            ipProvided = 1;
            i++; 
        }
    }

    if (!ipProvided || !numberProvided) {
        fprintf(stderr, "Error: Missing required arguments\n");
        return 1;
    }

    char *token = strtok(ip, ":");
    if (token == NULL || !isValidIpAddress(token)) {
        fprintf(stderr, "Error: Invalid IP address\n");
        return 1;
    }

    printf("recver ip: %s\n", ip);
    int recver_port = atoi(strtok(NULL, ":"));
    printf("recver port: %d\n", recver_port);

    const int TEST_TIME = number;

    // read recver ip from argv

    int encoder_socket, recver_socket;
    struct sockaddr_un encoder_addr;
    struct sockaddr_in recver_addr;

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

    if(connect(encoder_socket, (struct sockaddr *)&encoder_addr, sizeof(encoder_addr)) == -1){
        perror("Error: connect to encoder fail");
        exit(1);
    }

    /*connect to recver*/
    recver_socket = socket(AF_INET, SOCK_STREAM, 0);

    recver_addr.sin_family = AF_INET;
    recver_addr.sin_port = htons(recver_port);

    if(inet_pton(AF_INET, ip, &recver_addr.sin_addr) <= 0) {
        fprintf(stderr,"Invalid ip address/ Address not supported \n");
        return -1;
    }

    if(connect(recver_socket, (struct sockaddr *)&recver_addr, sizeof(recver_addr)) == -1){
        perror("Error connect to recver: ");
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
            // write(recver_socket, response, strlen(response));
            send(recver_socket, response, strlen(response), 0);
            // wait for recver ok
            // read(recver_socket, response, MAX_PACKET_LEN);
            recv(recver_socket, response, MAX_PACKET_LEN, 0);

            // write(recver_socket, SemanticRL_example[i], strlen(SemanticRL_example[i]));
        }
    }

    printf("%d/%d exceed 10ms\n", exceed_10ms, TEST_TIME);
    
    close(encoder_socket);
    close(recver_socket);
    free(response);
}