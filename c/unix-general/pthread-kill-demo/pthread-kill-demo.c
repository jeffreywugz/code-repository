#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/signal.h>

void panic(const char* msg)
{
        perror(msg);
        exit(-1);
}

void *threadfunc(void *parm)
{
        pthread_t self;
        self = pthread_self();
        printf("thread: %x\n", (unsigned)self);
        pause();
        return NULL;
}

void sighand(int signo)
{
        pthread_t self;
        self = pthread_self();
        printf("catch signal in thread: %x\n", (unsigned)self);
        printf("signo: %d\n", signo);
}

int main(int argc, char **argv)
{
        int rc;
        struct sigaction actions;
        pthread_t thread;

        memset(&actions, 0, sizeof(actions));
        sigemptyset(&actions.sa_mask);
        actions.sa_flags = 0;
        actions.sa_handler = sighand;

        rc = sigaction(SIGALRM,&actions,NULL);
        if(rc < 0)panic("sigaction");
 
        rc = pthread_create(&thread, NULL, threadfunc, NULL);
        if(rc < 0)panic("pthread_create");

        sleep(1);               /* sleep essential */
        rc = pthread_kill(thread, SIGALRM);
        if(rc < 0)panic("pthread_kill");

        rc = pthread_join(thread, NULL);
        if(rc < 0)panic("pthread_join");
        return 0;
}
