#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
//#include <sync.h>
const char* usages = "%s array_size n_loop n_thread\n";

#define mfence() __asm__("mfence");
#define compile_barrier() asm volatile("" ::: "memory")
#define arrlen(x) (sizeof(x)/sizeof(x[0]))

int64_t* big_array_;
int64_t* big_array2_;
int64_t array_size_;
volatile int stop_read_;
volatile int64_t* volatile pint_;
volatile int64_t loop_;
pthread_barrier_t barrier_;
int64_t do_read_check(int64_t n)
{
  int64_t n_err = 0;
  int64_t loop = 1;
  while(1)
  {
    pthread_barrier_wait(&barrier_);
    loop = loop_;
    if (loop < 0)
    {
      return n_err;
    }
    while(!stop_read_)
    {
      if (*pint_ != loop)
      {
        n_err++;
        fprintf(stderr, "n_error=%ld, loop=%ld", n_err, loop);
      }
    }
    pthread_barrier_wait(&barrier_);
  }
}

int64_t do_write(int64_t n)
{
  int64_t *p = NULL;
  int64_t loop = 0;
  int64_t i = 0;
  for(loop = 0; loop < n; loop++)
  {
    loop_++;
    pint_ = &loop_;
    stop_read_ = 0;
    printf("loop=%ld\n", loop_);
    pthread_barrier_wait(&barrier_);
    for(i = 0; i < array_size_; i++)
    {
      p = big_array_ + i;
      *p = loop_;
      compile_barrier();
      pint_ = p;
    }
    stop_read_ = 1;
    pthread_barrier_wait(&barrier_);
  }
  loop_ = -1;
  pthread_barrier_wait(&barrier_);
  return 0;
}

typedef void*(*pthread_handler_t)(void*);
int64_t test_out_of_order(int64_t array_size, int64_t n_loop, int64_t n_thread)
{
  int64_t total_err = 0, thread_err = 0;
  int64_t i = 0;
  pthread_t thread[128];
  printf("test_out_of_order(array_size=%ld, n_loop=%ld, n_thread=%ld)\n", array_size, n_loop, n_thread);
  pthread_barrier_init(&barrier_, NULL, n_thread + 1);
  if (array_size <= 0 || n_loop <= 0 || n_thread <= 0 || n_thread + 1 > arrlen(thread))
  {
    total_err = -EINVAL;
  }
  else if (NULL == (big_array_ = malloc(sizeof(int64_t) * array_size)))
  {
    total_err = -ENOMEM;
  }
  else
  {
    memset(big_array_, 0, sizeof(int64_t) * array_size);
    array_size_ = array_size;
    pthread_create(&thread[0], NULL, (pthread_handler_t)do_write, (void*)n_loop);
    for(i = 1; i < n_thread + 1; i++)
    {
      pthread_create(&thread[i], NULL, (pthread_handler_t)do_read_check, (void*)n_loop);
    }
    for(i = 0; i < n_thread + 1; i++)
    {
      pthread_join(thread[i], (void**)(&thread_err));
      total_err += thread_err;
    }
  }
  if (NULL != big_array_)
  {
    free(big_array_);
  }
  pthread_barrier_destroy(&barrier_);
  return total_err;
}

int main(int argc, char *argv[])
{
  int err = 0;
  if (argc != 4)
  {
    err = -EINVAL;
    printf(usages, argv[0]);
  }
  else
  {
    int64_t array_size = atoll(argv[1]);
    int64_t n_loop = atoll(argv[2]);
    int64_t n_thread = atoll(argv[3]);
    printf("n_err:%ld\n", test_out_of_order(array_size, n_loop, n_thread));
  }
  return err;
}
