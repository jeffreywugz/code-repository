#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>

void panic(const char* msg)
{
        perror(msg);
        exit(-1);
}


#define MAX_MSG_LEN 256
#define SOCK_PATH "file.unix-domain"

int unix_server()
{
        int sock, conn, err, len;
        char msg_buf[MAX_MSG_LEN+1];
        struct sockaddr_un local, remote;
        int local_len, remote_len;
        int max_request_clients = 10;

        sock = socket(AF_UNIX, SOCK_STREAM, 0);
        if(sock < 0)panic("socket");

        local.sun_family = AF_UNIX;
        strcpy(local.sun_path, SOCK_PATH);
        local_len = strlen(local.sun_path) + sizeof(local.sun_family);
        unlink(local.sun_path);
        
        err = bind(sock, (struct sockaddr *)&local, local_len);
        if(err)panic("bind");
        err = listen(sock, max_request_clients);
        if(err)panic("listen");
    
        conn = accept(sock, (struct sockaddr*)&remote, (socklen_t*)&remote_len);
        if(conn<0)panic("accept");
        len = recv(conn, msg_buf, MAX_MSG_LEN, 0);
        if(len <= 0)panic("recv");
        msg_buf[len] = '\0';
        close(conn);

        close(sock);
    
        printf("bytes: %d payload: %s\n", len, msg_buf);

        return 0;
}

int unix_client()
{
        int sock, len, err;
        struct sockaddr_un remote;
        int remote_len;
        char msg_buf[] = "hello...";

        sock = socket(AF_UNIX, SOCK_STREAM, 0);
        if(sock < 0)panic("socket");
        
        remote.sun_family = AF_UNIX;
        strcpy(remote.sun_path, SOCK_PATH);
        remote_len = strlen(remote.sun_path) + sizeof(remote.sun_family);

        err = connect(sock, (struct sockaddr*)&remote, remote_len);
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
                unix_server();
        } else {
                sleep(1);
                unix_client();
        }           
	return 0;
}
