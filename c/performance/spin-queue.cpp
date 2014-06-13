#include "profile.h"
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <assert.h>
#include <locale.h>

#define CACHE_ALIGN_SIZE 64
#define CACHE_ALIGNED __attribute__((aligned(CACHE_ALIGN_SIZE)))
#define FAA(addr, x) __sync_fetch_and_and(addr, x)
#define CAS(addr, oldv, newv) __sync_bool_compare_and_swap(addr, oldv, newv)
#define FAS(addr, v) __sync_lock_test_and_set(addr, v)
#define AL(x) __atomic_load_n((x), __ATOMIC_SEQ_CST)
//#define AL(x) (*(x))
#define PAUSE() asm("pause;\n")

class FastQueue
{
public:
  FastQueue(): push_(0), pop_(0), capacity_(0), items_(NULL) {}
  ~FastQueue(){}
  int init(uint64_t capacity, void** items) {
    int err = 0;
    capacity_ = capacity;
    pos_mask_ = capacity - 1;
    memset(items, 0, sizeof(void*) * capacity);
    items_ = items;
    return err;
  }
  int destroy() {
    int err = 0;
    items_ = NULL;
    return err;
  }
  inline int64_t idx(int64_t x) { return x & (pos_mask_); }
  inline int push(void* p) {
    int err = 0;
    uint64_t push = FAA(&push_, 1);
    void** pdata = items_ + idx(push);
    //while(NULL != AL(pdata) || !CAS(pdata, NULL, p))
    while(!CAS(pdata, NULL, p))
    {
      PAUSE();
    }
    return err;
  }
  inline int pop(void*& p) {
    int err = 0;
    uint64_t pop = FAA(&pop_, 1);
    void** pdata = items_ + idx(pop);
    while(NULL == AL(pdata) || NULL == (p = FAS(pdata, NULL)))
      //while(NULL == (p = FAS(pdata, NULL)))
    {
      PAUSE();
    }
    return err;
  }
  int64_t remain() const { return push_ - pop_; }
private:
  volatile uint64_t push_ CACHE_ALIGNED;
  volatile uint64_t pop_ CACHE_ALIGNED;
  uint64_t capacity_ CACHE_ALIGNED;
  uint64_t pos_mask_ CACHE_ALIGNED;
  void** items_;
};

class NormalQueue
{
public:
  NormalQueue(): count_(0), push_(0), pop_(0), capacity_(0), items_(NULL) {}
  ~NormalQueue(){}
  int init(uint64_t capacity, void** items) {
    int err = 0;
    capacity_ = capacity;
    memset(items, 0, sizeof(void*) * capacity);
    items_ = items;
    return err;
  }
  int destroy() {
    int err = 0;
    items_ = NULL;
    return err;
  }
  int64_t idx(int64_t x) { return x & (capacity_ - 1); }
  int push(void* p) {
    int err = -EAGAIN;
    uint64_t count = 0;
    while((count = AL(&count_)) < capacity_
          && !CAS(&count_, count, count + 1))
      ;
    if (count < capacity_)
    {
      uint64_t push = FAA(&push_, 1);
      void** pdata = items_ + idx(push);
      while(!CAS(pdata, NULL, p))
      {
        PAUSE();
      }
      err = 0;
    }
    return err;
  }
  int pop(void*& p) {
    int err = -EAGAIN;
    uint64_t count = 0;
    while((count = AL(&count_)) > 0
          && !CAS(&count_, count, count - 1))
      ;
    if (count > 0)
    {
      uint64_t pop = FAA(&pop_, 1);
      void** pdata = items_ + idx(pop);
      while(NULL == (p = FAS(pdata, NULL)))
      {
        PAUSE();
      }
      err = 0;
    }
    return err;
  }
  int64_t remain() const { return push_ - pop_; }
private:
  uint64_t count_ CACHE_ALIGNED;
  uint64_t push_ CACHE_ALIGNED;
  uint64_t pop_ CACHE_ALIGNED;
  uint64_t capacity_ CACHE_ALIGNED;
  void** items_;
};

class NormalQueue2
{
public:
  NormalQueue2(): push_(0), pop_(0), capacity_(0), items_(NULL) {}
  ~NormalQueue2(){}
  int init(uint64_t capacity, void** items) {
    int err = 0;
    capacity_ = capacity;
    memset(items, 0, sizeof(void*) * capacity);
    items_ = items;
    return err;
  }
  int destroy() {
    int err = 0;
    items_ = NULL;
    return err;
  }
  int64_t idx(int64_t x) { return x & (capacity_ - 1); }
  int push(void* p) {
    int err = -EAGAIN;
    uint64_t push = 0;
    while((push = AL(&push_)) < AL(&pop_) + capacity_)
    {
      if (!CAS(&push_, push, push + 1))
      {
        PAUSE();
      }
      else
      {
        err = 0;
        break;
        while(!CAS(items_ + idx(push), NULL, p))
        {
          PAUSE();
        }
        err = 0;
        break;
      }
    }
    return err;
  }
  int pop(void*& p) {
    int err = -EAGAIN;
    uint64_t pop = 0;
    while((pop = AL(&pop_)) < AL(&push_))
    {
      if (!CAS(&pop_, pop, pop + 1))
      {
        PAUSE();
      }
      else
      {
        err = 0;
        break;
        void** pdata = items_ + idx(pop);
        while(NULL == (p = AL(pdata)) || !CAS(pdata, p, NULL))
        {
          PAUSE();
        }
        err = 0;
        break;
      }
    }
    return err;
  }
  int64_t remain() const { return push_ - pop_; }
private:
  uint64_t count_ CACHE_ALIGNED;
  uint64_t push_ CACHE_ALIGNED;
  uint64_t pop_ CACHE_ALIGNED;
  uint64_t capacity_ CACHE_ALIGNED;
  void** items_;
};

typedef NormalQueue Queue;
struct QueueCallable: public Callable
{
  Queue queue_;
  int64_t n_items_;
  int64_t n_push_done_;
  int64_t n_push_thread_;
  int64_t n_pop_thread_;
  QueueCallable(): n_items_(0), n_push_done_(0), n_push_thread_(0), n_pop_thread_(0) {}
  virtual ~QueueCallable() {}
  QueueCallable& set(int64_t n_items, int64_t queue_len_shift, int64_t n_push_thread, int64_t n_pop_thread) {
    fprintf(stderr, "queue_callable(n_items=%ld, queue_len=%ld)\n", n_items, 1LL<<queue_len_shift);
    n_items_ = n_items;
    n_push_thread_ = n_push_thread;
    n_pop_thread_ = n_pop_thread;
    queue_.init((1<<queue_len_shift), (void**)(malloc(sizeof(void*) * (1<<queue_len_shift))));
    return *this;
  }
  int call(pthread_t thread, int64_t idx) {
    int err = 0;
    fprintf(stdout, "worker[%ld] run\n", idx);
    if (idx < n_push_thread_)
    {
      for(int64_t i = 0; i < n_items_; i++) {
        while (0 != queue_.push((void*)(i+1))){
          //fprintf(stderr, "push(i=%ld):ERROR\n", i);
        }
      }
      __sync_fetch_and_add(&n_push_done_, 1);
    }
    else
    {
      void* p = NULL;
      for(int64_t i = 0; i < n_items_; i++) {
        while (0 != queue_.pop(p)){
          //fprintf(stderr, "pop(i=%ld):ERROR\n", i);
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
  int64_t queue_len_shift = 20;
  setlocale(LC_ALL, "");
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
  return err;
}
