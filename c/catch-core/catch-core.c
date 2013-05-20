#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>

void sighandler(int signum)
{
  printf("Process %d got signal %d\n", getpid(), signum);
  kill(getpid(), SIGSTOP);
}

int main()
{
  signal(SIGSEGV, sighandler);
  //kill(getpid(), SIGSEGV);
  char * p = 0;
  *p = 0;
  return 0;
}
