#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>

void panic(const char* msg)
{
        perror(msg);
        exit(-1);
}

int counter = 0;
void inc(void* args)
{
        int i;
        printf("thread 0x%lx start.\n", pthread_self());
        while(1){
                for(i = 0; i < (1<<14); i++)
                        ;
                counter++;
        }
}

#define N 8
int main()
{
        int i;
        pthread_t thread[N];

        int pid = fork();
        if(pid > 0){
                exit(0);
        }
                
        printf("start.\n");
        for(i = 0; i < N; i++)
                if(pthread_create(thread+i, NULL, (void*)inc, NULL) != 0)
                        panic("pthread_create");
  
        /* for(i = 0; i < N; i++) */
        /*         if (pthread_join(thread[i], NULL) != 0) */
        /*                 panic("pthread_join"); */
 
        sleep(1);
        printf("counter: %d\n", counter);
        return 0; 
}

