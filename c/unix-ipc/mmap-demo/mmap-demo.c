#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>

void panic(const char* msg)
{
        perror(msg);
        exit(-1);
}

int main()
{
        pid_t pid;
        char* buf;
        int buf_len = 20;
        const char* file_name = "file.mmap";
        int fd;
        fd = open(file_name, O_RDWR | O_CREAT);
        if(fd<0)panic("open");
        if(ftruncate(fd, buf_len)<0)panic("ftruncate");
        buf = mmap(0, buf_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if(buf == MAP_FAILED)panic("mmap");
        close(fd);
        unlink(file_name);
        
        pid = fork();
        if(pid < 0)panic("fork");
        if(pid > 0){
                strcpy(buf, "hello...");
        } else {
                usleep(1000);
                printf("child: %s\n", buf);
        }
                           
        munmap(buf, buf_len);
	return 0;
}
