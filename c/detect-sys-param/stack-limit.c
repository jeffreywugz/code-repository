#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int64_t get_thread_stack_size()
{
  int64_t stack_size = 0;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_getstacksize (&attr, &stack_size);
  return stack_size;
}

int64_t test_stack_limit(int64_t depth, char* base)
{
  char top = 0;
  char buf[1<<10];
  fprintf(stderr, "stack depth=%ld, size=%ld\n", depth, base - &top);
  return test_stack_limit(depth+1, base);
}

void* _test_stack_limit(void* arg)
{
  char base = 0;
  test_stack_limit(0, &base);
}

int main(int argc, char *argv[])
{
  pthread_t thread;
  pthread_attr_t attr;
  int64_t n = 0;
  n = argc > 1 ? atoi(argv[1]): 20;
  n = (1 << (n > 0 ? n: 10));
  printf("%ld\n", n);
  printf("thread stack size=%ld\n", get_thread_stack_size());
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, n);
  pthread_create(&thread, &attr, _test_stack_limit, NULL);
  pthread_join(thread, NULL);
  return 0;
}
