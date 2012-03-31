#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <errno.h>
#include <pthread.h>

#define array_len(x) (sizeof(x)/sizeof(x[0]))
int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

#define profile(expr) ({                                                \
      int64_t old_us = 0;                                               \
      int64_t new_us = 0;                                               \
      int64_t result = 0;                                               \
      old_us = get_usec();                                              \
      result = expr;                                                    \
      new_us = get_usec();                                              \
      printf("%s=>%ld in %ldms\n", #expr, result, new_us - old_us);     \
      new_us - old_us; })


class Queue
{
  public:
    Queue(): capacity_(0), items_(NULL), front_(0), rear_(0) {}
    ~Queue() {}

    int init(void** items, int64_t capacity) {
      int err = 0;
      if (NULL == items || 0 >= capacity)
      {
        err = -EINVAL;
      }
      else
      {
        items_ = items;
        capacity_ = capacity;
      }
      return err;
    }

    int push(void* p) {
      int64_t rear = rear_;
      if (NULL == items_ || rear&1 || (front_>>1) + capacity_ <= (rear>>1) || !__sync_bool_compare_and_swap(&rear_, rear, rear+1))
        return -EAGAIN;
      items_[(rear>>1)%capacity_] = p;
      rear_++;
      return 0;
    }

    int pop(void*& p) {
      int64_t front = front_;
      if (NULL == items_ || front&1 || front+1 >= rear_ || !__sync_bool_compare_and_swap(&front_, front, front+1))
        return -EAGAIN;
      p = items_[(front>>1)%capacity_];
      front_++;
      return 0;
    }
  private:
    int64_t capacity_;
    void** items_;
    volatile int64_t front_;
    volatile int64_t rear_;
};

#define ref_header volatile int64_t ref_
struct ref_t
{
  ref_header;
};

class SlicePool
{
  public:
    SlicePool() {}
    ~SlicePool() {}
  public:
    int init(void** items, char* buf, int64_t n_items, int64_t item_size) {
      int err = 0;
      if (NULL == items || NULL == buf || 0 >= item_size || 0 >= n_items)
      {
        err = -EINVAL;
      }
      else if (0 != (err = queue_.init(items, n_items)))
      {}
      for(int64_t i = 0; 0 == err && i < n_items; i++)
      {
        ((ref_t*)(buf + i*item_size))->ref_ = 0;
        err = queue_.push(buf + i*item_size);
      }
      return err;
    }

    int alloc(void*& p) {
      int err = 0;
      while(-EAGAIN == (err = queue_.pop(p)))
        ;
      if (0 == err)
      {
        ((ref_t*)p)->ref_ = 1;
      }
      return err;
    }

    int ref(void* p) const {
      int err = 0;
      int64_t ref = __sync_fetch_and_add(&((ref_t*)p)->ref_, 1);
      if (ref <= 0)
      {
        err = -EINVAL;
      }
      return err;
    }

    int deref(void* p) {
      int err = 0;
      int64_t ref = __sync_add_and_fetch(&((ref_t*)p)->ref_, -1);
      if (ref < 0)
      {
        err = -EINVAL;
      }
      else if (ref == 0)
      {
        while(-EAGAIN == (err = queue_.push(p)))
              ;
      }
      return err;
    }
  private:
    char* buf_;
    Queue queue_;
};

struct Node
{
  ref_header;
  int64_t value_;
};

SlicePool slice_pool;
int64_t do_alloc_and_free(int64_t n)
{
  int64_t n_err = 0;
  int err = 0;
  void* p = NULL;
  for(int64_t i = 0; i < n; i++) {
    if (0 != (err = slice_pool.alloc(p)))
      n_err++;
    else if (0 != (err = slice_pool.deref(p)))
      n_err++;
  }
  return n_err;
}

typedef void*(*pthread_handler_t)(void*);
int64_t test_queue(int64_t n)
{
  int err = 0;
  pthread_t thread[1<<2];
  char* buf = (char*)malloc(n * sizeof(void*) + n * sizeof(Node));
  assert(buf);
  err = slice_pool.init((void**)buf, buf + n*sizeof(void*), n, sizeof(Node));
  assert(0 == err);
  for(int64_t i = 0; i < array_len(thread); i++)
    pthread_create(thread+i, NULL, (pthread_handler_t)do_alloc_and_free, (void*)n);
  for(int64_t i = 0; i < array_len(thread); i++){
    pthread_join(thread[i], NULL);
  }
  if(buf)free(buf);
  return 0;
}

int main(int argc, char *argv[])
{
  int64_t n = (argc > 1)? atoi(argv[1]): 0;
  n = 1<<(n?:10);
  printf("n=%ld\n", n);
  profile(test_queue(n));
  return 0;
}
