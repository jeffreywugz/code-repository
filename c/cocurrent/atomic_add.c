#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <sys/time.h>
int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

#define profile(expr) ({                                                \
      int64_t old_us = 0;                                               \
      int64_t new_us = 0;                                               \
      int64_t result = 0;                                               \
      old_us = get_usec();                                              \
      result = expr;                                                    \
      new_us = get_usec();                                              \
      printf("%s=>%ld in %ldms\n", #expr, result, new_us - old_us);     \
      new_us - old_us; })

int64_t n = 0;
int inc(int64_t* x)
{
  for(int64_t i = 0; i < n; i++) {
    (*x)++;
  }
  return 0;
}

int volatile_inc(int64_t* x)
{
  volatile int64_t* px = x;
  for(int64_t i = 0; i < n; i++) {
    (*px)++;
  }
  return 0;
}
int64_t delta = (1LL<<34) | 1;
int fetch_and_add(int64_t* x)
{
  for(int64_t i = 0; i < n; i++) {
    __sync_fetch_and_add(x, delta);
  }
  return 0;
}

int locked_inc(int64_t* x)
{
  for(int64_t i = 0; i < n; i++) {
    __asm__ volatile ( "lock incq %0"
                       :"=m" (*x)
                       : "m" (*x));
  }
  return 0;
}

typedef void*(*pthread_handler_t)(void*);
typedef int (*inc_handler_t)(int64_t*);
int64_t test_atomic_add(inc_handler_t handler)
{
  pthread_t thread1;
  pthread_t thread2;
  int64_t shift = 0;
  char p[1024];
  int64_t start = 0;
  for (shift = 0; shift < 128; shift += 4) {
    start = get_usec();
    *(int64_t*)(p+shift) = 0;
    pthread_create(&thread1, NULL, (pthread_handler_t)handler, p+shift);
    pthread_create(&thread2, NULL, (pthread_handler_t)handler, p+shift);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    printf("delta=%lx, shift=%ld, n_err=%ld, time=%ld\n", delta, shift, 2*n*delta - *(int64_t*)(p+shift), get_usec()-start);
  }
  return 0;
}

int main(int argc, char *argv[])
{
  int64_t x = 0;
  n = argc > 1 ? atoi(argv[1]): 10;
  n = (1 << (n > 0 ? n: 10));
  printf("%ld\n", n);
  //profile(test_atomic_add(inc));
  //profile(test_atomic_add(volatile_inc));
  profile(test_atomic_add(fetch_and_add));
  //profile(test_atomic_add(locked_inc));
  return 0;
}
