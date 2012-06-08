#include <errno.h>
#include <stdint.h>
#include <pthread.h>
#include <stdio.h>

int64_t assert_errno(int64_t idx)
{
  printf("pthread[%ld]: errno=%ld\n", idx, &errno);
  return 0;
}

typedef void*(*pthread_handler_t)(void*);
int test_errno(const int64_t n_threads, const int64_t n_iter)
{
  int64_t total_err = 0;
  int64_t n_err = 0;
  pthread_t threads[n_threads];
  for(int64_t i = 0; i < n_threads; i++) {
    pthread_create(threads + i, NULL, (pthread_handler_t)assert_errno, i);
  }
  for(int64_t i = 0; i < n_threads; i++) {
    n_err = 0;
    pthread_join(threads[i], (void**)&n_err);
    total_err += n_err;
  }
  return total_err;
}

int main()
{
  int64_t n_threads = 16;
  int64_t n_iter = 1<<10;
  printf("test_errno(n_threads=%ld, n_iter=%ld): total_err=%d\n", n_threads, n_iter, test_errno(n_threads, n_iter));
  return 0;
}
