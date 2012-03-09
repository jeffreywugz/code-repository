#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>

int64_t n = 0;
extern int64_t n_err;
void add();
void check();

int64_t writer(void* arg)
{
  for(int64_t i = 0; i < n; i++) {
    add();
  }
  return 0;
}

int64_t read_and_check(void* arg)
{
  for(int64_t i = 0; i < n; i++)
    check();
  return 0;
}

typedef void*(*pthread_handler_t)(void*);

int64_t test_out_of_order()
{
  pthread_t thread1;
  pthread_t thread2;
  pthread_create(&thread1, NULL, (pthread_handler_t)writer, NULL);
  pthread_create(&thread2, NULL, (pthread_handler_t)read_and_check, NULL);

  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);
  return n_err;
}

int main(int argc, char *argv[])
{
  n = argc > 1 ? atoi(argv[1]): 10;
  n = (1LL << (n > 0 ? n: 10));
  printf("n=%ld\n", n);
  printf("n_err:%ld\n", test_out_of_order());
  return 0;
}
