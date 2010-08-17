#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <semaphore.h>

void panic(const char* msg)
{
        perror(msg);
        exit(-1);
}

int main()
{
        pid_t pid;
        const char* file_name = "file.sem";
        sem_t* sem;
        int value = 1;
        sem = sem_open(file_name, O_CREAT, S_IRWXU, value);
        if(sem == SEM_FAILED)panic("sem_open");
        sem_unlink(file_name);
        
        pid = fork();
        if(pid < 0)panic("fork");
        if(pid > 0){
                sleep(1);
                sem_post(sem);
        } else {
                sem_wait(sem);
                printf("get it!\n");
                sem_wait(sem);
                printf("get it!\n");
        }
                           
        sem_close(sem);
	return 0;
}
