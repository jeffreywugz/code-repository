#include "profile.h"
#include <assert.h>
#include <string.h>

#define CACHE_ALIGN_SIZE 64
#define CACHE_ALIGNED __attribute__((aligned(CACHE_ALIGN_SIZE)))
#define PAUSE() asm("pause\n")

struct SeqLock
{
  static const int64_t N_THREAD = 128;
  static const int64_t N_THREAD_MASK = N_THREAD - 1;
  int64_t done_seq_[N_THREAD][CACHE_ALIGN_SIZE/sizeof(int64_t)];
  int64_t do_seq_ CACHE_ALIGNED;
  SeqLock(): do_seq_(0) { memset(done_seq_, 0, sizeof(done_seq_)); }
  int64_t lock()
  {
    int64_t seq = __sync_add_and_fetch(&do_seq_, 1);
    volatile int64_t& last_done_seq = done_seq_[(seq-1) & N_THREAD_MASK][0];
    while(last_done_seq < seq - 1)
      PAUSE();
    __sync_synchronize();
    return seq;
  }
  int64_t unlock(int64_t seq)
  {
    volatile int64_t& this_done_seq = done_seq_[seq & N_THREAD_MASK][0];
    this_done_seq = seq;
    __sync_synchronize();
    return seq;
  }
};

struct IncSeq
{
  SeqLock seq_;
  volatile int64_t value_ CACHE_ALIGNED;
  IncSeq(): seq_(), value_(0){}
  ~IncSeq(){}
  int64_t next(int64_t& value)
  {
    int64_t seq = seq_.lock();
    if (value > value_)
    {
      value_ = value;
    }
    else
    {
      value = ++value_;
    }
    return seq_.unlock(seq);
  }
};

struct SeqCallable: public Callable
{
  IncSeq seq_;
  int64_t n_items_;
  int64_t queue_len_shift_;
  SeqCallable& set(int64_t n_items, int64_t queue_len_shift) {
    fprintf(stderr, "queue_callable(n_items=%ld, queue_len=%ld)\n", n_items, 1LL<<queue_len_shift);
    n_items_ = n_items;
    return *this;
  }
  int call(pthread_t thread, int64_t idx) {
    int err = 0;
    int64_t cur_ts = 0;
    fprintf(stdout, "worker[%ld] run\n", idx);
    for(int64_t i = 0; i < n_items_; i++) {
      cur_ts = get_usec();
      seq_.next(cur_ts);
    }
    return err;
  }
};

int main(int argc, char** argv)
{
  int err = 0;
  BaseWorker worker;
  SeqCallable callable CACHE_ALIGNED;
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
    profile(worker.set_thread_num(n_thread).par_do(&callable.set(n_items, queue_len_shift)), n_items * n_thread);
  }
}
