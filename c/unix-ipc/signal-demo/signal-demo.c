#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>

void panic(const char* msg)
{
        perror(msg);
        exit(-1);
}

void int_handler(int sig_num)
{
        printf("catch it!\n");
}

int main()
{
        int sig_num = SIGINT;
        struct sigaction new_action, old_action;
        new_action.sa_handler = int_handler;
        sigemptyset(&new_action.sa_mask);
        new_action.sa_flags = 0;
        if(sigaction(sig_num, &new_action, &old_action))panic("sigaction");
        while(1);
	return 0;
}
