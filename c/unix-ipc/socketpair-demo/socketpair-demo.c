#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>

#define MAX_MSG_LEN 256

void panic(const char* msg)
{
        perror(msg);
        exit(-1);
}

int main()
{
        pid_t pid;
        int fd[2];
        char msg_send_buf[] = "hello...";
        char msg_recv_buf[MAX_MSG_LEN];
        int msg_len = sizeof(msg_send_buf);
        int read_bytes, write_bytes;
        
        if(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == -1)
                panic("socketpair");
        
        pid = fork();
        if(pid < 0)panic("fork");
        if(pid > 0){
                close(fd[1]);
                read_bytes = read(fd[0], msg_recv_buf, msg_len);
                if(read_bytes != msg_len)panic("read");
                printf("recv: %s\n", msg_recv_buf);
                close(fd[0]);
        } else {
                close(fd[0]);
                printf("send: %s\n", msg_send_buf);
                write_bytes = write(fd[1], msg_send_buf, msg_len);
                if(write_bytes != msg_len)panic("write");
                close(fd[1]);
        }           
	return 0;
}
