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

int tcp_server()
{
    int sock, conn, err, len;
    char msg_buf[MAX_MSG_LEN+1];
    struct sockaddr_in saddr, fromaddr;
    int fromaddr_len = sizeof(fromaddr);
    int max_request_clients = 10;

    sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if(sock < 0)panic("socket");

    saddr.sin_family = AF_INET;
    saddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    saddr.sin_port = htons(61321);
    memset(&saddr.sin_zero, 0, sizeof(saddr.sin_zero));
    
    err = bind(sock, (struct sockaddr *)&saddr, sizeof(saddr));
    if(err)panic("bind");
    err = listen(sock, max_request_clients);
    if(err)panic("listen");
    
    conn = accept(sock, (struct sockaddr*)&fromaddr, (socklen_t*)&fromaddr_len);
    if(conn<0)panic("accept");
    len = recv(conn, msg_buf, MAX_MSG_LEN, 0);
    if(len <= 0)panic("recv");
    msg_buf[len] = '\0';
    close(conn);

    close(sock);
    
    printf("bytes: %d payload: %s\n", len, msg_buf);
    printf("addr: %s port: %d",
           inet_ntoa(fromaddr.sin_addr), ntohs(fromaddr.sin_port));

    return 0;
}

int tcp_client()
{
    int sock, len, err;
    char msg_buf[] = "hello...";
    struct sockaddr_in saddr;

    sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if(sock < 0)panic("socket");

    saddr.sin_family = AF_INET;
    saddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    saddr.sin_port = htons(61321);
    memset(&saddr.sin_zero, 0, sizeof(saddr.sin_zero));

    err = connect(sock, (struct sockaddr*)&saddr, sizeof(saddr));
    if(err < 0)panic("connect");
    
    len = send(sock, msg_buf, sizeof(msg_buf), 0);
    if(len <= 0)panic("send");

    close(sock);
    
    return 0;
}

int main()
{
        pid_t pid;
        pid = fork();
        if(pid < 0)panic("fork");
        if(pid >0){
                tcp_server();
        } else {
                sleep(1);
                tcp_client();
        }           
	return 0;
}
