#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#define MAX_MSG_LEN 256

void panic(const char* msg)
{
        perror(msg);
        exit(-1);
}

int main()
{
        pid_t pid;
        const char* fifo_name = "file.fifo";
        int fd;
        char msg_send_buf[] = "hello...";
        char msg_recv_buf[MAX_MSG_LEN];
        int msg_len = sizeof(msg_send_buf);
        int read_bytes, write_bytes;

        if(mkfifo(fifo_name, S_IRWXU) < 0)panic("mkfifo");

        pid = fork();
        if(pid < 0)panic("fork");
        if(pid > 0){
                fd = open(fifo_name, O_RDONLY);
                unlink(fifo_name);
                if(fd < 0)panic("open");
                read_bytes = read(fd, msg_recv_buf, msg_len);
                if(read_bytes != msg_len)panic("read");
                printf("recv: %s\n", msg_recv_buf);
                close(fd);
        } else {
                fd = open(fifo_name, O_WRONLY);
                if(fd < 0)panic("open");
                printf("send: %s\n", msg_send_buf);
                write_bytes = write(fd, msg_send_buf, msg_len);
                if(write_bytes != msg_len)panic("write");
                close(fd);
        }           
	return 0;
}
