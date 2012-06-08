#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

const char my_interp[] __attribute__((section(".interp")))
    = "/lib64/ld-linux-x86-64.so.2";

void debug_loop()
{
  while(1) {
    pause();
    // inspect process status here using gdb
  }
}

typedef void*(*pthread_handler_t)(void*);
void __attribute__((constructor)) so_init()
{
  pthread_t thread;
  printf("in so init\n");
  pthread_create(&thread, NULL, (pthread_handler_t)debug_loop, NULL);
}

int mystart(int argc, char** argv)
{
  printf("a shared object demo\n");
  exit(0);
}
