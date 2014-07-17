#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#define AX_SUCCESS 0
#define AX_INVALID_ARGUMENT -2
#define AX_NOT_INIT -6
#define AX_EAGAIN -23
#define UNUSED(v) ((void)(v))
#define FAA(x, i) __sync_fetch_and_add((x), (i))
#define CAS(x, ov, nv) __sync_bool_compare_and_swap((x), (ov), (nv))
#define VCAS(x, ov, nv) __sync_val_compare_and_swap((x), (ov), (nv))
#define FAS(x, ov) __sync_lock_test_and_set((x), (ov))
#define AL(x) __atomic_load_n((x), __ATOMIC_SEQ_CST)
#define AS(x, v) __atomic_store_n((x), (v), __ATOMIC_SEQ_CST)
#define MBR() __sync_synchronize()
#define PAUSE() asm("pause;\n")

#define CACHE_ALIGN_SIZE 64
#define CACHE_ALIGNED __attribute__((aligned(CACHE_ALIGN_SIZE)))

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

volatile int64_t __next_tid __attribute__((weak));
inline int64_t itid()
{
  static __thread int64_t tid = -1;
  return tid < 0? (tid = __sync_fetch_and_add(&__next_tid, 1)): tid;
}

struct TCCounter
{
  enum {MAX_THREAD_NUM=256};
  struct Item
  {
    uint64_t value_;
  } CACHE_ALIGNED;
  TCCounter(){ memset(items_, 0, sizeof(items_)); }
  ~TCCounter(){}
  int64_t inc(int64_t delta=1) { return __sync_fetch_and_add(&items_[itid() % MAX_THREAD_NUM].value_, delta); }
  int64_t value() const{
    int64_t sum = 0;
    int64_t thread_count = __next_tid;
    int64_t slot_count = (thread_count < MAX_THREAD_NUM)? thread_count: MAX_THREAD_NUM;
    for(int64_t i = 0; i < slot_count; i++)
    {
      fprintf(stderr, "thread[%ld] = %ld\n", i, items_[i].value_);
      sum += items_[i].value_;
    }
    return sum;
  }
  Item items_[MAX_THREAD_NUM];
};

struct Callable
{
  Callable(): stop_(false) {}
  virtual ~Callable() {}
  virtual void stop() { stop_ = true; }
  virtual int call(pthread_t thread, int64_t idx) = 0;
  virtual void report(int64_t time_us) = 0;
  bool stop_ CACHE_ALIGNED;
};

template<typename T>
class OneLoopCallable: public Callable, public T
{
  public:
    OneLoopCallable(){}
    virtual ~OneLoopCallable(){}
    virtual int call(pthread_t thread, int64_t idx) {
      while(!stop_) {
        if (0 == this->do_once(idx))
        {
          counter_.inc(1);
        }
      }
      return 0;
    }
    void report(int64_t time_us) {
      fprintf(stdout, "time=%ldus, ops=%'ld\n", time_us, 1000000L * counter_.value()/time_us);
    }
  protected:
    TCCounter counter_;
};

typedef void*(*pthread_handler_t)(void*);
class BaseWorker
{
  public:
    static const int64_t MAX_N_THREAD =1024;
    struct WorkContext
    {
      WorkContext() : callable_(NULL), idx_(0) {}
      ~WorkContext() {}
      WorkContext& set(Callable* callable, int64_t idx) {
        callable_ = callable;
        idx_ = idx;
        return *this;
      }
      Callable* callable_;
      pthread_t thread_;
      int64_t idx_;
    };
  public:
    BaseWorker(): n_thread_(0) {}    
    ~BaseWorker(){}
  public:
    BaseWorker& set_thread_num(int64_t n){ n_thread_ = n; return *this; }
    int start(Callable* callable, int64_t idx=-1) {
      int err = 0;
      for(int64_t i = 0; i < n_thread_; i++){
        if (idx > 0 && idx != i)continue;
        fprintf(stderr, "worker[%ld] start.\n", i);
        pthread_create(&ctx_[i].thread_, NULL, (pthread_handler_t)do_work, (void*)(&ctx_[i].set(callable, i)));
      }
      return err;
    }

    int wait(int64_t idx=-1) {
      int err = 0;
      int64_t ret = 0;
      for(int64_t i = 0; i < n_thread_; i++) {
        if (idx > 0 && idx != i)continue;
        pthread_join(ctx_[i].thread_, (void**)&ret);
        if (ret != 0) {
          fprintf(stderr, "thread[%ld] => %ld\n", i, ret);
        } else {
          fprintf(stderr, "thread[%ld] => OK.\n", i);
        }
      }
      return err;
    }

    static int do_work(WorkContext* ctx) {
      int err = 0;
      if (NULL == ctx || NULL == ctx->callable_) {
        err = -EINVAL;
      } else {
        err = ctx->callable_->call(ctx->thread_, ctx->idx_);
      }
      return err;
    }
    int par_do(Callable* callable, int64_t time_us) {
      int err = 0;
      if (0 != (err = start(callable)))
      {
        fprintf(stderr, "start()=>%d\n", err);
      }
      else
      {
        usleep(time_us);
        callable->stop();
        if (0 != (err = wait()))
        {
          fprintf(stderr, "wait()=>%d\n", err);
        }
        else
        {
          fprintf(stderr, "par_do finished.\n");
          callable->report(time_us);
        }
      }
      return err;
    }
  protected:
    int64_t n_thread_;
    WorkContext ctx_[MAX_N_THREAD];
};

#if 0
struct SimpleCallable: public Callable
{
  int64_t n_items_;
  SimpleCallable& set(int64_t n_items) {
    n_items_ = n_items;
    return *this;
  }
  int call(pthread_t thread, int64_t idx) {
    int err = 0;
    fprintf(stdout, "worker[%ld] run\n", idx);
    if (idx % 2)
      err = -EPERM;
    return err;
  }
};

int main(int argc, char** argv)
{
  int err = 0;
  BaseWorker worker;
  SimpleCallable callable;
  int64_t n_thread = 0;
  int64_t n_items = 0;
  if (argc != 3)
  {
    err = -EINVAL;
    fprintf(stderr, "n_thread=1 time_limit=3 %s n_thread n_item\n", argv[0]);
  }
  else
  {
    n_thread = atoll(argv[1]);
    n_items = atoll(argv[2]);
    profile(worker.set_thread_num(n_thread).par_do(&callable.set(n_items)), n_items);
  }
}
#endif
