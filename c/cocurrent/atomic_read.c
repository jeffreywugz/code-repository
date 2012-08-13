#include <stdio.h>
#include <stdlirb.h>
#include <stdint.h>
#include <pthread.h>

int64_t n = 0;
const int64_t x1 = 0x00000000ffffffff;
const int64_t x2 = 0xffffffff00000000;
int set1(int64_t* x)
{
  for(int64_t i = 0; i < n; i++) {
    *x = x1;
  }
  return 0;
}

int set2(int64_t* x)
{
  for(int64_t i = 0; i < n; i++) {
    *x = x2;
  }
  return 0;
}

int64_t check(int64_t* x)
{
  int n_err = 0;
  int64_t tx = 0;
  for(int i = 0; i < n; i++) {
    tx = *x;
    if (tx != x1 && tx != x2)
      n_err++;
  }
  return n_err;
}

typedef void*(*pthread_handler_t)(void*);
int64_t test_atomic_read()
{
  pthread_t thread1;
  pthread_t thread2;
  pthread_t thread3;
  int64_t n_err = 0;
  int64_t shift = 0;
  char p[1024];
  for (shift = 0; shift < 1024; shift += 4) {
    *(int64_t*)(p+shift) = x1;
    pthread_create(&thread1, NULL, (pthread_handler_t)set1, p + shift);
    pthread_create(&thread2, NULL, (pthread_handler_t)set2, p + shift);
    pthread_create(&thread3, NULL, (pthread_handler_t)check, p + shift);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    pthread_join(thread3, (void**)&n_err);
    printf("shift:%ld, n_err: %ld\n", shift, n_err);
  }
  return n_err;
}

int main(int argc, char *argv[])
{
  n = argc > 1 ? atoi(argv[1]): 10;
  n = (1 << (n > 0 ? n: 10));
  printf("%ld\n", n);
  printf("n_err: %ld\n", test_atomic_read());
  return 0;
}
