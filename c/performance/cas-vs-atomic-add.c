#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <sys/time.h>

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

#define profile(expr, n) { \
  int64_t start = get_usec(); \
  expr;\
  int64_t end = get_usec();\
  printf("%s: 1000000*%ld/%ld=%ld\n", #expr, n, end - start, 1000000 * n / (end - start)); \
}

volatile int64_t gx = 0;
int do_cas(const int64_t n)
{
  int64_t x = 0;
  for(int64_t i = 0; i < n;) {
    x = gx;
    if(__sync_bool_compare_and_swap(&gx, x, x+1))
    {
      i++;
    }
  }
  return x;
}

int do_add(const int64_t n)
{
  int64_t x = 0;
  for(int64_t i = 0; i < n; i++)
    __sync_add_and_fetch(&gx, 1);
  return x;
}

int do_loop(const int64_t n)
{
  int64_t x = 0;
  for(int64_t i = 0; i < n; i++)
    x ^= i;
  return x;
}

typedef void*(*pthread_handler_t)(void*);
int par_do(pthread_handler_t do_work, const int64_t n_thread, const int64_t n_items)
{
  int err = 0;
  pthread_t threads[n_thread];
  for(int64_t i = 0; i < n_thread; i++)
    pthread_create(threads + i, NULL, (pthread_handler_t)do_work, (void*)(n_items/n_thread));
  for(int64_t i = 0; i < n_thread; i++)
    pthread_join(threads[i], NULL);
  return err;
}

int main(int argc, char** argv)
{
  int err = 0;
  int64_t n_thread = 0;
  int64_t n_items = 0;
  if (argc != 3)
  {
    err = -EINVAL;
    fprintf(stderr, "%s n_thread n_item\n", argv[0]);
  }
  else
  {
    n_thread = atoll(argv[1]);
    n_items = atoll(argv[2]);
    profile(par_do((pthread_handler_t)do_cas, n_thread, n_items), n_items);
    profile(par_do((pthread_handler_t)do_add, n_thread, n_items), n_items);
    profile(par_do((pthread_handler_t)do_loop, n_thread, n_items), n_items);
  }
}
