#include "profile.h"
#include <string.h>
#include <semaphore.h>
#include <time.h>
#include <assert.h>

#define CACHE_ALIGN_SIZE 64
#define CACHE_ALIGNED __attribute__((aligned(CACHE_ALIGN_SIZE)))
#define CAS(addr, oldv, newv) __sync_bool_compare_and_swap(addr, oldv, newv)
class Queue
{
  public:
    struct Item
    {
      volatile uint32_t seq_;
      volatile sem_t sem_;
      void* data_;
    };
  public:
    Queue(): push_(0), pop_(0), pos_mask_(0), items_(NULL) {}
    ~Queue(){}
    int init(uint64_t len_bits, Item* items) {
      int err = 0;
      pos_mask_ = (1<<len_bits) - 1;
      memset(items, 0, sizeof(Item) * (1<<len_bits));
      for(int64_t i = 0; 0 == err && i < (1<<len_bits); i++) {
        err = sem_init((sem_t*)&items[i].sem_, 0, 0);
      }
      if (0 == err) {
        items_ = items;
      }
      return err;
    }
    int destroy() {
      int err = 0;
      for(int64_t i = 0; NULL != items_ && 0 == err && i < (pos_mask_ + 1); i++) {
        err = sem_destroy((sem_t*)&items_[i].sem_);
      }
      items_ = NULL;
      return err;
    }
    int push(void* data) {
      int err = 0;
      uint64_t seq = __sync_fetch_and_add(&push_, 1);
      Item* pi = items_ + (seq & pos_mask_);
      if (0 != (err = sem_post((sem_t*)&pi->sem_)))
      {
        err = errno;
      }
      else
      {
        while(!CAS(&pi->seq_, 0, 1))
          ;
        pi->data_ = data;
        pi->seq_++;
      }
      return err;
    }
    int pop(void*& data, const struct timespec* end_time) {
      int err = 0;
      uint64_t seq = __sync_fetch_and_add(&pop_, 1);
      Item* pi = items_ + (seq & pos_mask_);
      if (0 != (err = sem_timedwait((sem_t*)&pi->sem_, end_time)))
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
    queue_.init(queue_len_shift, (Queue::Item*)(malloc(sizeof(Queue::Item) * (1<<queue_len_shift))));
    return *this;
  }
  int call(pthread_t thread, int64_t idx) {
    int err = 0;
    void* p = NULL;
    fprintf(stdout, "worker[%ld] run\n", idx);
    timespec end_time;
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
        clock_gettime(CLOCK_REALTIME, &end_time);
        end_time.tv_sec++;
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
