#include "profile.h"
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <sys/time.h>
#include <assert.h>

#define CACHE_ALIGN_SIZE 64
#define CACHE_ALIGNED __attribute__((aligned(CACHE_ALIGN_SIZE)))
#define CAS(addr, oldv, newv) __sync_bool_compare_and_swap(addr, oldv, newv)
#define futex(...) syscall(SYS_futex,__VA_ARGS__)

#define NS_PER_SEC 1000000000
timespec* calc_remain_timespec(timespec* remain, const timespec* end_time)
{
  timespec* ret = NULL;
  timespec now;
  if (NULL == end_time || NULL == remain)
  {}
  else if (0 != clock_gettime(CLOCK_REALTIME, &now))
  {
    remain->tv_sec = 0;
    remain->tv_nsec = 0;
    ret = remain;
  }
  else
  {
    remain->tv_sec = end_time->tv_sec - now.tv_sec;
    remain->tv_nsec = end_time->tv_nsec - now.tv_nsec;
    if (remain->tv_nsec < 0)
    {
      remain->tv_sec--;
      remain->tv_nsec + NS_PER_SEC;
    }
    ret = remain;
  }
  return ret;
}

timespec* calc_abs_time(timespec* ts, const int64_t time_ns)
{
  if (NULL == ts)
  {}
  else if (0 != (clock_gettime(CLOCK_REALTIME, ts)))
  {}
  else
  {
    ts->tv_nsec += time_ns;
    if (ts->tv_nsec > NS_PER_SEC)
    {
      ts->tv_sec += ts->tv_sec/NS_PER_SEC;
      ts->tv_nsec %= NS_PER_SEC;
    }
  }
  return ts;
}

static int futex_wake(int* p, int val)
{
  int err = 0;
  if (0 != futex(p, FUTEX_WAKE_PRIVATE, val, NULL, NULL, 0))
  {
    err = errno;
  }
  return err;
}

static int futex_wait(int* p, int val, const timespec* end_time)
{
  int err = 0;
  timespec remain;
  if (0 != futex(p, FUTEX_WAIT_PRIVATE, val, calc_remain_timespec(&remain, end_time), NULL, 0))
  {
    err = errno;
  }
  return err;
}

int dec_if_gt0(volatile int* p)
{
  int x = 0;
  while((x = *p) > 0 && !CAS(p, x, x - 1))
    ;
  return x;
}

int inc_if_le0(volatile int* p)
{
  int x = 0;
  while((x = *p) <= 0 && !CAS(p, x, x + 1))
    ;
  return x;
}

struct FCounter
{
  int32_t val_;
  int32_t n_waiters_;
  FCounter(): val_(0), n_waiters_(0) {}
  ~FCounter() {}
  int32_t inc(const timespec* end_time)
  {
    int err = 0;
    int32_t val = 0;
    if ((val = inc_if_le0(&val_)) <= 0)
    {}
    else
    {
      __sync_add_and_fetch(&n_waiters_, 1);
      while(true)
      {
        if (ETIMEDOUT == (err = futex_wait(&val_, val, end_time)))
        {
          val = __sync_fetch_and_add(&val_, 1);
          break;
        }
        if ((val = inc_if_le0(&val_)) <= 0)
        {
          break;
        }
      }
      __sync_add_and_fetch(&n_waiters_, -1);
    }
    if (n_waiters_ > 0)
    {
      futex_wake(&val_, INT_MAX);
    }
    return val;
  }

  int32_t dec(const timespec* end_time)
  {
    int err = 0;
    int32_t val = 0;
    if ((val = dec_if_gt0(&val_)) > 0)
    {}
    else
    {
      __sync_add_and_fetch(&n_waiters_, 1);
      while(true)
      {
        if (ETIMEDOUT == (err = futex_wait(&val_, val, end_time)))
        {
          val = __sync_fetch_and_add(&val_, -1);
          break;
        }
        if ((val = dec_if_gt0(&val_)) > 0)
        {
          break;
        }
      }
      __sync_add_and_fetch(&n_waiters_, -1);
    }
    if (n_waiters_ > 0)
    {
      futex_wake(&val_, INT_MAX);
    }
    return val;
  }
};

class Queue
{
  public:
    struct Item
    {
      FCounter counter_;
      void* volatile data_;
      int push(void* data, const timespec* end_time) {
        int err = 0;
        if (counter_.inc(end_time) != 0)
        {
          err = -EAGAIN;
        }
        else
        {
          while(!CAS(&data_, NULL, data))
            ;
        }
        return err;
      }
      int pop(void*& data, const timespec* end_time) {
        int err = 0;
        if (counter_.dec(end_time) != 1)
        {
          err = -EAGAIN;
        }
        else
        {
          while(!(NULL != (data = data_) && CAS(&data_, data, NULL)))
            ;
        }
        return err;
      }
    };
  public:
    Queue(): push_(0), pop_(0), pos_mask_(0), items_(NULL) {}
    ~Queue(){}
    int init(uint64_t pos_mask, Item* items) {
      int err = 0;
      pos_mask_ = pos_mask;
      memset(items, 0, sizeof(Item) * (pos_mask + 1));
      items_ = items;
      return err;
    }
    int destroy() {
      int err = 0;
      items_ = NULL;
      return err;
    }
    int push(void* data, const timespec* end_time) {
      int err = 0;
      uint64_t seq = __sync_fetch_and_add(&push_, 1);
      Item* pi = items_ + (seq & pos_mask_);
      err = pi->push(data, end_time);
      return err;
    }
    int pop(void*& data, const struct timespec* end_time) {
      int err = 0;
      uint64_t seq = __sync_fetch_and_add(&pop_, 1);
      Item* pi = items_ + (seq & pos_mask_);
      err = pi->pop(data, end_time);
      return err;
    }
    int64_t remain() const { return push_ - pop_; }
  private:
    volatile uint64_t push_ CACHE_ALIGNED;
    volatile uint64_t pop_ CACHE_ALIGNED;
    uint64_t pos_mask_ CACHE_ALIGNED;
    Item* items_;
};

struct QueueCallable: public Callable
{
  Queue queue_;
  int64_t n_items_;
  int64_t n_item_pushed_;
  int64_t n_push_done_;
  int64_t n_push_thread_;
  int64_t n_pop_thread_;
  QueueCallable(): n_items_(0), n_item_pushed_(0), n_push_done_(0), n_push_thread_(0), n_pop_thread_(0) {}
  ~QueueCallable() {}
  QueueCallable& set(int64_t n_items, int64_t queue_len_shift, int64_t n_push_thread, int64_t n_pop_thread) {
    fprintf(stderr, "queue_callable(n_items=%ld, queue_len=%ld)\n", n_items, 1<<queue_len_shift);
    n_items_ = n_items;
    n_push_thread_ = n_push_thread;
    n_pop_thread_ = n_pop_thread;
    queue_.init((1<<queue_len_shift)-1, (Queue::Item*)(malloc(sizeof(Queue::Item) * (1<<queue_len_shift))));
    return *this;
  }
  int call(pthread_t thread, int64_t idx) {
    int err = 0;
    int64_t timeout_ns = 10 * 1000000;
    fprintf(stdout, "worker[%ld] run\n", idx);
    timespec end_time;
    if (idx < n_push_thread_)
    {
      for(int64_t i = 0; i < n_items_; i++) {
        if (0 != (err = queue_.push((void*)(i+1), calc_abs_time(&end_time, timeout_ns))))
        {
          fprintf(stderr, "push(idx=%ld, i=%ld)=>%d\n", idx, i, err);
        }
        //__sync_fetch_and_add(&n_item_pushed_, 1);
      }
      //__sync_fetch_and_add(&n_push_done_, 1);
    }
    else
    {
      void* p = NULL;
      for(int64_t i = 0; i < n_items_; i++) {
        if (0 != (err = queue_.pop(p, calc_abs_time(&end_time, timeout_ns))))
        {
          fprintf(stderr, "pop(idx=%ld, i=%ld)=>%d\n", idx, i, err);
        }
      }
    }
    return err;
  }
};

int main(int argc, char** argv)
{
  int err = 0;
  BaseWorker worker;
  QueueCallable callable;
  int64_t n_push_thread = 0;
  int64_t n_pop_thread = 0;
  int64_t n_items = 0;
  int64_t queue_len_shift = 2;
  if (argc != 4)
  {
    err = -EINVAL;
    fprintf(stderr, "%s n_push_thread n_pop_thread n_item\n", argv[0]);
  }
  else
  {
    n_push_thread = atoll(argv[1]);
    n_pop_thread = atoll(argv[2]);
    n_items = atoll(argv[3]);
    callable.set(n_items, queue_len_shift, n_push_thread, n_pop_thread);
    worker.set_thread_num(n_push_thread + n_pop_thread);
    profile(worker.par_do(&callable), n_items * n_push_thread);
  }
}
