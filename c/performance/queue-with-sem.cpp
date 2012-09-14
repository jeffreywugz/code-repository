#include "profile.h"
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

int decrement_if_positive(int* p)
{
  int x = 0;
  while((x = *p) > 0 && !__sync_bool_compare_and_swap(p, x, x - 1))
    ;
  return x;
}

struct my_sem_t
{
  int32_t val_;
  int32_t nwaiters_;
};

static int futex_post(volatile int* addr)
{
  int err = 0;
  int val = 0;
  __sync_fetch_and_add(addr, 1);
  if (val < 0)
    futex(addr, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
  return err;
}

static int futex_wait(my_sem_t* p, const timespec* timeout)
{
  int err = 0;
  if (decrement_if_positive(p->val_) > 0)
    return err;
  __sync_fetch_and_add(p->nwaiters_, 1);
  while(decrement_if_positive(p->val_) <= 0
        && 0 != (err = futex(p->val_, FUTEX_WAIT_PRIVATE, *p->val_, timeout, NULL, 0)
                 && ETIMEDOUT != err)
        ;
  __sync_fetch_and_add(p->nwaiters_, 1);
  return err;
}

class Queue
{
  public:
    struct Item
    {
      volatile uint32_t seq_;
      volatile int32_t sem_;
      void* data_;
      char buf[0] CACHE_ALIGNED;
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
    int push(void* data) {
      int err = 0;
      uint64_t seq = __sync_fetch_and_add(&push_, 1);
      Item* pi = items_ + (seq & pos_mask_);
      if (0 != (err = futex_post(&pi->sem_)))
      {
        err = errno;
      }
      else
      {
        while(!CAS(&pi->seq_, 0, 1))
          ;
        pi->data_ = data;
        pi->seq_ = 2;
      }
      return err;
    }
    int pop(void*& data, const struct timespec* timeout) {
      int err = 0;
      uint64_t seq = __sync_fetch_and_add(&pop_, 1);
      Item* pi = items_ + (seq & pos_mask_);
      if (0 != (err = futex_wait(&pi->sem_, timeout)))
      {
        err = errno; // 可能返回ETIMEDOUT
      }
      else
      {
        while(!CAS(&pi->seq_, 2, 3))
          ;
        data = pi->data_;
        pi->seq_ = 0;
      }
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
  int64_t n_push_done_;
  int64_t n_push_thread_;
  int64_t n_pop_thread_;
  QueueCallable(): n_items_(0), n_push_done_(0), n_push_thread_(0), n_pop_thread_(0) {}
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
    void* p = NULL;
    fprintf(stdout, "worker[%ld] run\n", idx);
    timespec end_time;
    end_time.tv_sec = 0;
    end_time.tv_nsec = 10 * 1000 * 1000;
    if (idx < n_push_thread_)
    {
      for(int64_t i = 0; i < n_items_; i++) {
        queue_.push(p);
      }
      __sync_fetch_and_add(&n_push_done_, 1);
    }
    else
    {
      for(; n_push_done_ != n_push_thread_; ) {
        //clock_gettime(CLOCK_REALTIME, &end_time);
        queue_.pop(p, &end_time);
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
  int64_t queue_len_shift = 10;
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
