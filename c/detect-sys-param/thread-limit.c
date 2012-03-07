#define _BSD_SOURCE
#include <unistd.h>
#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

bool stop = false;
void* thread_run(void* arg)
{
  while(!stop)
    usleep(10 * 1000);
}

int64_t get_thread_limit(int64_t n)
{
  int err = 0;
  int64_t i = 0;
  pthread_t* thread = malloc(n * sizeof(pthread_t));
  assert(thread);
  for(i = 0; 0 == err && i < n; i++) {
    err = pthread_create(thread + i, NULL, thread_run, NULL);
    fprintf(stderr, "thread[%ld] create: err=%d\n", i, err);
  }
  stop = true;
  for(int64_t j = 0; j < i; j++) {
    pthread_join(thread[j], NULL);
  }
  free(thread);
  return i;
}

int main(int argc, char *argv[])
{
  int i = (argc > 1)? atoi(argv[1]): 0;
  i = (i == 0)? 3: i;
  printf("thread number limit: %ld\n", get_thread_limit(i));
  return 0;
}
