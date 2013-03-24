#include <stdint.h>
#include "profile.h"
#include <assert.h>

#define CACHE_ALIGN_SIZE 64
#define CACHE_ALIGNED __attribute__((aligned(CACHE_ALIGN_SIZE)))
#define PAUSE() asm("pause\n")

namespace types
{
  struct uint128_t
  {
    uint64_t lo;
    uint64_t hi;
  }
  __attribute__ (( __aligned__( 16 ) ));
}

bool cas128( volatile types::uint128_t * src, types::uint128_t cmp, types::uint128_t with )
{
  bool result;
  __asm__ __volatile__
    (
      "\n\tlock cmpxchg16b %1"
      "\n\tsetz %0\n"
      : "=q" ( result )
        , "+m" ( *src )
        , "+d" ( cmp.hi )
        , "+a" ( cmp.lo )
      : "c" ( with.hi )
        , "b" ( with.lo )
      : "cc"
      );
  return result;
}

#define CAS128(src, cmp, with) cas128((types::uint128_t*)(src), *((types::uint128_t*)&(cmp)), *((types::uint128_t*)&(with)))
struct IncSeq
{
  int64_t seq_;
  int64_t ts_;
  IncSeq(): seq_(0), ts_(0) {}
  ~IncSeq(){}
  bool next(int64_t& ts)
  {
    IncSeq old = *this;
    IncSeq tmp;
    tmp.seq_ = old.seq_ + 1;
    tmp.ts_ = ts > old.ts_? ts: old.ts_ + 1;
    return CAS128(this, old, tmp);
  }
}__attribute__ (( __aligned__( 16 ) ));

struct CasCallable: public Callable
{
  IncSeq seq_;
  int64_t n_items_;
  int64_t queue_len_shift_;
  CasCallable& set(int64_t n_items) {
    fprintf(stderr, "cas_callable(n_items=%ld)\n", n_items);
    n_items_ = n_items;
    return *this;
  }
  int call(pthread_t thread, int64_t idx) {
    int err = 0;
    int64_t cur_ts = 0;
    fprintf(stdout, "worker[%ld] run\n", idx);
    for(int64_t i = 0; i < n_items_; i++) {
      cur_ts = get_usec();
      while(!seq_.next(cur_ts))
        ;
    }
    return err;
  }
};

int main(int argc, char** argv)
{
  int err = 0;
  BaseWorker worker;
  CasCallable callable;
  int64_t n_thread = 0;
  int64_t n_items = 0;
  if (argc != 3)
  {
    err = -EINVAL;
    fprintf(stderr, "%s n_thread n_item\n", argv[0]);
  }
  else
  {
    n_thread = atoll(argv[1]);
    n_items = atoll(argv[2]);
    profile(worker.set_thread_num(n_thread).par_do(&callable.set(n_items)), n_items * n_thread);
  }
}
