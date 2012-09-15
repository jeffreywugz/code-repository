#include "profile.h"
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <assert.h>

#define CACHE_ALIGN_SIZE 64
#define CACHE_ALIGNED __attribute__((aligned(CACHE_ALIGN_SIZE)))
#define CAS(addr, oldv, newv) __sync_bool_compare_and_swap(addr, oldv, newv)

class Queue
{
  public:
  public:
    Queue(): push_(0), pop_(0), pos_mask_(0), items_(NULL) {}
    ~Queue(){}
    int init(uint64_t pos_mask, void** items) {
      int err = 0;
      pos_mask_ = pos_mask;
      memset(items, 0, sizeof(void*) * (pos_mask + 1));
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
      void* volatile * pi = items_ + (seq & pos_mask_);
      while(!CAS(pi, NULL, data))
        ;
      return err;
    }
    int pop(void*& data) {
      int err = 0;
      uint64_t seq = __sync_fetch_and_add(&pop_, 1);
      void* volatile * pi = items_ + (seq & pos_mask_);
      while(NULL == (data = *pi) || !CAS(pi, data, NULL))
        ;
      return err;
    }
    int64_t remain() const { return push_ - pop_; }
  private:
    volatile uint64_t push_ CACHE_ALIGNED;
    volatile uint64_t pop_ CACHE_ALIGNED;
    uint64_t pos_mask_ CACHE_ALIGNED;
    void** items_;
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
    queue_.init((1<<queue_len_shift)-1, (void**)(malloc(sizeof(void*) * (1<<queue_len_shift))));
    return *this;
  }
  int call(pthread_t thread, int64_t idx) {
    int err = 0;
    fprintf(stdout, "worker[%ld] run\n", idx);
    timespec end_time;
    if (idx < n_push_thread_)
    {
      for(int64_t i = 0; i < n_items_; i++) {
        if (0 != queue_.push((void*)(i+1)))
          fprintf(stderr, "push(i=%ld):ERROR\n", i);
      }
      __sync_fetch_and_add(&n_push_done_, 1);
    }
    else
    {
      void* p = NULL;
      for(int64_t i = 0; i < n_items_; i++) {
        if (0 != queue_.pop(p))
          fprintf(stderr, "pop(i=%ld):ERROR\n", i);
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
  int64_t queue_len_shift = 8;
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
