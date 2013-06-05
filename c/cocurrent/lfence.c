#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

//#define mfence() __asm__("mfence");
#define mfence()

int64_t check()
{
  volatile static uint64_t s = 0;
  int64_t err = 0;
  __sync_fetch_and_add(&s, 1);
  if (s >= 2)
    err = 1;
  __sync_fetch_and_add(&s, -1);
  return err;
}

volatile int x = 0;
int64_t some_array[32];
volatile int y = 0;
int64_t funcA(int64_t n)
{
  int64_t n_err = 0;
  for(int64_t i = 0; i < n; i++) {
    x++;
    y++;
  }
  return n_err;
}

int64_t funcB(int64_t n)
{
  int64_t n_err = 0;
  int64_t
  for(int64_t i = 0; i < n; i++) {
  }
  return n_err;
}

typedef void*(*pthread_handler_t)(void*);
int64_t test_out_of_order(int64_t n)
{
  int64_t n_err1 = 0, n_err2 = 0;
  pthread_t thread1, thread2;
  pthread_create(&thread1, NULL, (pthread_handler_t)funcA, (void*)n);
  pthread_create(&thread2, NULL, (pthread_handler_t)funcB, (void*)n);
  pthread_join(thread1, (void**)&n_err1);
  pthread_join(thread2, (void**)&n_err2);
  return n_err1 + n_err2;
}

int main(int argc, char *argv[])
{
  int64_t n = 0;
  n = argc > 1 ? atoi(argv[1]): 10;
  n = (1LL << (n > 0 ? n: 10));
  printf("n=%ld\n", n);
  printf("n_err:%ld\n", test_out_of_order(n));
  return 0;
}
