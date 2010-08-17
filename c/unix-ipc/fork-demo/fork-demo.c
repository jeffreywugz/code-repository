#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

void panic(const char* msg)
{
        perror(msg);
        exit(-1);
}

int main()
{
        pid_t pid;
        pid = fork();
        if(pid < 0)panic("fork");
        if(pid >0)
                printf("hello from parent!\n");
        else
                printf("hello from child!\n");
                           
	return 0;
}
