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

template<typename T>
struct SeqLockedType
{
  SeqLockedType(): seq__(0) {}
  ~SeqLockedType() {}
  volatile int64_t seq__;
  int update(T* that)
  {
    int err = 0;
    int64_t seq = seq__;
    if (seq&1 || !__sync_bool_compare_and_swap(&seq__, seq, ++seq))
    {
      err = -EAGAIN;
    }
    else
    {
      that->seq__ = seq;
      *(T*)this = *that;
    }
    if (0 == err && !__sync_bool_compare_and_swap(&seq__, seq, ++seq))
    {
      err = -EAGAIN;
    }
    return err;
  }

  int read(T* that) const
  {
    int err = 0;
    int64_t seq = seq__;
    if (seq&1)
    {
      err = -EAGAIN;
    }
    else
    {
      *that = *(T*)this;
    }
    if (0 == err && seq__ != seq)
    {
      err = -EAGAIN;
    }
    return 0;
  }
};

struct Node: public SeqLockedType<Node>
{
  Node(): x(0), y(0) {}
  ~Node() {}
  int64_t x;
  int64_t y;
};

struct Node gnode;
int64_t n = 0;
int64_t read_write_with_lock(int64_t idx)
{
  int64_t n_err = 0;
  Node node;
  for(int64_t i = 0; i < n; i++){
    if (i&1) // write
    {
      node.x = idx * n + i;
      node.y = idx * n + i;
      gnode.update(&node);
    }
    else // read
    {
      if (0 == gnode.read(&node) && node.x != node.y)
      {
        n_err++;
      }
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
      gnode.x = idx * n + i;
      gnode.y = idx * n + i;
    }
    else // read gnode
    {
      if (gnode.x != gnode.y) // invalid read
      {
        n_err++;
      }
    }
  }
  return n_err;
}

enum {USE_SEQ_LOCK, NO_LOCK};
typedef void*(*pthread_handler_t)(void*);
int64_t test_seq_lock(int flag)
{
  int64_t n_err = 0;
  int64_t total_err = 0;
  pthread_t thread[1<<2];
  for(int64_t i = 0; i < array_len(thread); i++)
    pthread_create(thread + i, NULL, (pthread_handler_t)(flag == USE_SEQ_LOCK? read_write_with_lock: read_write_no_lock), (void*)i);
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
  profile(test_seq_lock(USE_SEQ_LOCK));
  return 0;
}
