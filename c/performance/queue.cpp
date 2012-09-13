#include "profile.h"
#include <assert.h>

#define CACHE_ALIGN_SIZE 64
#define CACHE_ALIGNED __attribute__((aligned(CACHE_ALIGN_SIZE)))

class Queue
{
  public:
    struct Item
    {
      volatile uint64_t seq_;
      volatile int32_t n_push_waiter_;
      volatile int32_t n_pop_waiter_;
      void* data_;
    };
  public:
    Queue(): push_(0), pop_(0), capacity_(0), len_bits_(0), pos_mask_(0), items_(NULL) {}
    ~Queue(){}
    static uint64_t get_pos(const uint64_t seq, const uint64_t mask, const uint64_t len_bits) {
#if 0
      const int64_t tride_bits = 4;
      return (seq & mask)>>tride_bits | seq<<(len_bits - tride_bits) & mask;
#else
      return seq & mask;
#endif
    }
    int init(uint64_t len_bits, Item* items) {
      int err = 0;
      Item* pitem = NULL;
      len_bits_ = len_bits;
      capacity_ = 1<<len_bits;
      pos_mask_ = (1<<len_bits) - 1;
      items_ = items;
      for(uint64_t i = 0; i < capacity_; i++) {
        pitem = &items_[get_pos(i, pos_mask_, len_bits_)];
        pitem->seq_ = (i<<1);
        pitem->n_push_waiter_ = 0;
        pitem->n_pop_waiter_ = 0;
      }
      return err;
    }
    int push(void* data) {
      int err = 0;
      uint64_t seq = __sync_fetch_and_add(&push_, 2);
      Item* pi = items_ + get_pos(seq>>1, pos_mask_, len_bits_);
      while(pi->seq_ != seq){
        ;
      }
      pi->data_ = data;
      pi->seq_++;
      return err;
    }
    int pop(void*& data) {
      int err = 0;
      uint64_t seq = __sync_fetch_and_add(&pop_, 2);
      Item* pi = items_ + get_pos(seq>>1, pos_mask_, len_bits_);
      while(pi->seq_ != seq + 1) {
        ;
      }
      data = pi->data_;
      pi->seq_ += (capacity_<<1)-1;
      return err;
    }
  private:
    volatile uint64_t push_ CACHE_ALIGNED;
    volatile uint64_t pop_ CACHE_ALIGNED;
    uint64_t capacity_ CACHE_ALIGNED;
    uint64_t len_bits_;
    uint64_t pos_mask_;
    Item* items_;
};

struct QueueCallable: public Callable
{
  Queue queue_;
  int64_t n_items_;
  QueueCallable& set(int64_t n_items, int64_t queue_len_shift) {
    fprintf(stderr, "queue_callable(n_items=%ld, queue_len=%ld)\n", n_items, 1<<queue_len_shift);
    n_items_ = n_items;
    queue_.init(queue_len_shift, (Queue::Item*)(malloc(sizeof(Queue::Item) * (1<<queue_len_shift))));
    return *this;
  }
  int call(pthread_t thread, int64_t idx) {
    int err = 0;
    void* p = NULL;
    fprintf(stdout, "worker[%ld] run\n", idx);
    if (idx % 2) {
      for(int64_t i = 0; i < n_items_; i++) {
        queue_.pop(p);
      }
    } else {
      for(int64_t i = 0; i < n_items_; i++) {
        queue_.push(p);
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
  int64_t n_thread = 0;
  int64_t n_items = 0;
  int64_t queue_len_shift = 10;
  if (argc != 3)
  {
    err = -EINVAL;
    fprintf(stderr, "%s n_thread n_item\n", argv[0]);
  }
  else
  {
    n_thread = atoll(argv[1]);
    n_items = atoll(argv[2]);
    profile(worker.set_thread_num(n_thread).par_do(&callable.set(n_items, queue_len_shift)), n_items * n_thread/2);
  }
}
