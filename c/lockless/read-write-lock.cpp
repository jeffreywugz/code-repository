#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>

#define array_len(x) (sizeof(x)/sizeof(x[0]))
int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

#define profile(expr) ({                        \
  int64_t old_us = 0;                                  \
  int64_t new_us = 0;                                  \
  int64_t result = 0;                                    \
  old_us = get_usec();                                   \
  result = expr;                                         \
  new_us = get_usec();                                                  \
  printf("%s=>%ld in %ldms\n", #expr, result, new_us - old_us);           \
  new_us - old_us; })

struct ReadWriteLock
{
  ReadWriteLock(): ref_(0) {}
  ~ReadWriteLock() {}
  volatile int64_t ref_;
  bool set_write_flag() {
    int64_t ref = ref_;
    return __sync_bool_compare_and_swap(&ref_, ref, ref | 1);
  }
  bool clear_write_flag() {
    int64_t ref = ref_;
    return __sync_bool_compare_and_swap(&ref_, ref, ref & ~1);
  }
  bool write_lock() {
    return __sync_bool_compare_and_swap(&ref_, 1, 3);
  }
  bool write_unlock() {
    return __sync_bool_compare_and_swap(&ref_, 3, 1);
  }
  bool read_lock() {
    int64_t ref = ref_;
    return 0 == (ref&1) && __sync_bool_compare_and_swap(&ref_, ref, ref + 2);
  }
  bool read_unlock() {
    int64_t ref = ref_;
    return __sync_bool_compare_and_swap(&ref_, ref, ref - 2);
  }
};

ReadWriteLock rw_lock;
int64_t n = 0;
int64_t x = 0;
int64_t y = 0;
int64_t read_write_with_lock(int64_t idx)
{
  int64_t n_err = 0;
  for(int64_t i = 0; i < n; i++){
    if (i&1) // write
    {
      while(!rw_lock.set_write_flag())
        ;
      while(!rw_lock.write_lock())
        ;
      x++;
      y++;
      while(!rw_lock.write_unlock())
        ;
      while(!rw_lock.clear_write_flag())
        ;
    }
    else // read
    {
      while(!rw_lock.read_lock())
        ;
      if (x != y)
        n_err++;
      while(!rw_lock.read_unlock())
        ;
    }
  }
  return n_err;
}

int64_t read_write_no_lock(int64_t idx)
{
  int64_t n_err = 0;
  for(int64_t i = 0; i < n; i++){
    if (i&1) // update gnode
    {
      x++;
      y++;
    }
    else // read gnode
    {
      if (x != y)
      {
        n_err++;
      }
    }
  }
  return n_err;
}

enum {USE_LOCK, NO_LOCK};
typedef void*(*pthread_handler_t)(void*);
int64_t test_seq_lock(int flag)
{
  int64_t n_err = 0;
  int64_t total_err = 0;
  pthread_t thread[1<<1];
  for(int64_t i = 0; i < array_len(thread); i++)
    pthread_create(thread + i, NULL, (pthread_handler_t)(flag == USE_LOCK? read_write_with_lock: read_write_no_lock), (void*)i);
  for(int64_t i = 0; i < array_len(thread); i++){
    pthread_join(thread[i], (void**)&n_err);
    total_err += n_err;
  }
  return n_err;
}

int main(int argc, char *argv[])
{
  int i = (argc > 1)? atoi(argv[1]): 0;
  n = 1<<(i?:10);
  printf("n=%ld\n", n);
  profile(test_seq_lock(NO_LOCK));
  profile(test_seq_lock(USE_LOCK));
  return 0;
}
