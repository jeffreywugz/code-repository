#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

void panic(const char* msg)
{
        perror(msg);
        exit(-1);
}


#define MAX_MSG_LEN 256

int udp_server()
{
    int sock, err, len;
    char msg_buf[MAX_MSG_LEN+1];
    struct sockaddr_in saddr, fromaddr;
    int fromaddr_len = sizeof(fromaddr);

    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if(sock < 0)panic("socket");

    saddr.sin_family = AF_INET;
    saddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    saddr.sin_port = htons(61321);
    memset(&saddr.sin_zero, 0, sizeof(saddr.sin_zero));
    
    err = bind(sock, (struct sockaddr *)&saddr, sizeof(saddr));
    if(err)panic("bind");
   
    len = recvfrom(sock, msg_buf, MAX_MSG_LEN-1, 0,
                   (struct sockaddr *)&fromaddr, (socklen_t*)&fromaddr_len);
    if(len <= 0)panic("recvfrom");
    msg_buf[len] = '\0';

    close(sock);
    
    printf("bytes: %d payload: %s\n", len, msg_buf);
    printf("addr: %s port: %d",
           inet_ntoa(fromaddr.sin_addr), ntohs(fromaddr.sin_port));

    return 0;
}

int udp_client()
{
    int sock, len;
    char msg_buf[] = "hello...";
    struct sockaddr_in saddr;

    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if(sock < 0)panic("socket");

    saddr.sin_family = AF_INET;
    saddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    saddr.sin_port = htons(61321);
    memset(&saddr.sin_zero, 0, sizeof(saddr.sin_zero));
    
    len = sendto(sock, msg_buf, sizeof(msg_buf), 0,
                   (struct sockaddr *)&saddr, sizeof(saddr));
    if(len <= 0)panic("sendto");

    close(sock);
    
    return 0;
}

int main()
{
        pid_t pid;
        pid = fork();
        if(pid < 0)panic("fork");
        if(pid >0){
                udp_server();
        } else {
                sleep(1);       /* make sure server can receive the packet */
                udp_client();
        }           
	return 0;
}
