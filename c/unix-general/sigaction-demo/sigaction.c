#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/signal.h>

void panic(const char* msg)
{
        perror(msg);
        exit(-1);
}

void sighand(int signo)
{
        printf("signo: %d\n", signo);
        return;
}

int main(int argc, char **argv)
{
        int rc;
        struct sigaction actions;

        memset(&actions, 0, sizeof(actions));
        sigemptyset(&actions.sa_mask);
        actions.sa_flags = 0;
        actions.sa_handler = sighand;

        rc = sigaction(SIGALRM, &actions, NULL);
        if(rc < 0)panic("sigaction");

        alarm(1);
        pause();
        
        return 0;
}
