#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>

int g_x = 0;
int core(){ return *(char*)(0) = 0; }

void paused_core()
{
  pause();
  core();
}

int fib(int x)
{
  if(x < 0)return -1;
  if(x <= 2)return 1;
  return fib(x-1) + fib(x-2);
}

void run_fib_loop()
{
  pause();
  for(int i = 0; i < 1<<2; i++){
    sleep(1);
    printf("fib(%d) = %d\n", i, g_x=fib(i));
  }
}

void sig_handler(int sig)
{
  printf("cat sig:%d\n", sig);
}

int main()
{
  pthread_t fib_thread, core_thread;
  signal(SIGUSR1, sig_handler);
  pthread_create(&fib_thread, NULL, (void* (*)(void*))run_fib_loop, NULL);
  pthread_create(&core_thread, NULL, (void* (*)(void*))paused_core, NULL);
  pthread_join(fib_thread, NULL);
  pthread_join(core_thread, NULL);
  return 0;
}
