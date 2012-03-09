#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int64_t get_thread_cond_limit(int64_t n)
{
  int err = 0;
  int64_t i = 0;
  pthread_cond_t* cond = malloc(n * sizeof(pthread_cond_t));
  assert(cond);
  for(i = 0; 0 == err && i < n; i++) {
    err = pthread_cond_init(cond + i, NULL);
    if (i % ((n/1000)+1) == 0)
      fprintf(stderr, "pthread_cond_init(seq=%ld)=>%d\n", i, err);
  }
  return i;
}

int main(int argc, char *argv[])
{
  int i = (argc > 1)? atoi(argv[1]): 0;
  i = (i == 0)? 3: i;
  printf("thread cond limit: %ld\n", get_thread_cond_limit(i));
  return 0;
}
