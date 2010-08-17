#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <pthread.h>

void panic(const char* msg)
{
        perror(msg);
        exit(-1);
}

void do_some_thing(const char* arg)
{
        printf("arg:%s\n",  arg);
}

int main()
{
        pthread_t thread;
        const char* arg="just arg";
 
        if(pthread_create(&thread, NULL, (void*)do_some_thing, (void*)arg) != 0)
                panic("pthread_create");
  
        if (pthread_join(thread, NULL) != 0)
                panic("pthread_join");
 
        return 0; 
}

