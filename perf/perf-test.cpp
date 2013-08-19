#include <locale.h>
#include "profile.h"

class MutexCallable: public OneLoopCallable
{
  public:
    MutexCallable() { pthread_mutex_init(&mutex_, NULL); }
    ~MutexCallable() { pthread_mutex_destroy(&mutex_); }
    int do_once(int64_t idx) {
      UNUSED(idx);
      pthread_mutex_lock(&mutex_);
      pthread_mutex_unlock(&mutex_);
      return 0;
    }
  private:
    pthread_mutex_t mutex_ CACHE_ALIGNED;
};


class SpinLockCallable: public OneLoopCallable
{
  public:
    SpinLockCallable() { pthread_spin_init(&spinlock_, NULL); }
    ~SpinLockCallable() { pthread_spin_destroy(&spinlock_); }
    int do_once(int64_t idx) {
      UNUSED(idx);
      pthread_spin_lock(&spinlock_);
      pthread_spin_unlock(&spinlock_);
      return 0;
    }
  private:
    pthread_spinlock_t spinlock_ CACHE_ALIGNED;
};

class TSICallable: public OneLoopCallable
{
  public:
    TSICallable() { pthread_key_create(&key_, NULL); }
    ~TSICallable() { pthread_key_delete(key_); }
    int do_once(int64_t idx) {
      if (NULL == pthread_getspecific(key_))
      {
        pthread_setspecific(key_, (void*)(idx + 1));
      }
      return 0;
    }
  private:
    pthread_key_t key_ CACHE_ALIGNED;
};

class AddCallable: public OneLoopCallable
{
  public:
    AddCallable() { value_ = 0; }
    ~AddCallable() {}
    int do_once(int64_t idx) {
      __sync_fetch_and_add(&value_, 1);
      return 0;
    }
  private:
    int64_t value_ CACHE_ALIGNED;
};

class CasCallable: public OneLoopCallable
{
  public:
    CasCallable() { value_ = 0; }
    ~CasCallable() {}
    int do_once(int64_t idx) {
      while(true)
      {
        int64_t value = value_;
        if (__sync_bool_compare_and_swap(&value_, value, value + 1))
        {
          break;
        }
        else
        {
          PAUSE();
        }
      }
      return 0;
    }
  private:
    volatile int64_t value_ CACHE_ALIGNED;
};

class MemcpyCallable: public OneLoopCallable
{
  public:
    MemcpyCallable(int64_t block_size, int64_t n_block):block_size_(block_size), n_block_(n_block) { src_ = (char*)malloc(block_size * n_block); dest_ = (char*)malloc(block_size * n_block); }
    ~MemcpyCallable() {
      if(src_){ free(src_); src_ = NULL; }
      if(dest_){ free(dest_); dest_ = NULL; }
    }
    int do_once(int64_t idx) {
      static __thread int64_t pos = 0;
      int64_t real_pos = (pos + idx * block_size_)% (block_size_*n_block_);
      memcpy(dest_ + real_pos, src_ + real_pos, block_size_);
      pos += block_size_ * 10240;
      return 0;
    }
  private:
    int64_t block_size_;
    int64_t n_block_;
    char* src_;
    char* dest_;
};

class GetUsCallable: public OneLoopCallable
{
  public:
    GetUsCallable(){}
    ~GetUsCallable(){}
  public:
    int do_once(int64_t idx) {
      struct timeval time_val;
      gettimeofday(&time_val, NULL);
      last_time_ = time_val.tv_sec*1000000 + time_val.tv_usec;
      return 0;
    }
  private:
    int64_t last_time_;
};

#include "fast-spinlock.h"
class FastSpinLockCallable: public OneLoopCallable
{
  public:
    FastSpinLockCallable() { }
    ~FastSpinLockCallable() {}
    int do_once(int64_t idx) {
      spinlock_.lock_and_unlock();
      return 0;
    }
  private:
    FastSpinLock spinlock_;
};

#include "cas128.h"
class Cas128Callable: public OneLoopCallable
{
  public:
    Cas128Callable() { cur_value_ = 0; }
    ~Cas128Callable() {}
    int do_once(int64_t idx) {
      while(true)
      {
        __int128_t old_value;
        __int128_t new_value;
        LOAD128(old_value, &cur_value_);
        new_value = old_value + 1;
        if (CAS128(&cur_value_, old_value, new_value))
        {
          break;
        }
        else
        {
          PAUSE();
        }
      }
      return 0;
    }
  private:
    volatile __int128_t cur_value_ CACHE_ALIGNED;
};

#define _cfg(k, v) getenv(k)?:v
#define _cfgi(k, v) atoll(getenv(k)?:v)
class Perf
{
  public:
    int profile(Callable* callable) {
      int64_t n_thread = _cfgi("n_thread", "1");
      int64_t time_limit = _cfgi("time_limit", "1") * 1000000;
      BaseWorker worker;
      return worker.set_thread_num(n_thread).par_do(callable, time_limit);
    }
    int mutex() {
      MutexCallable callable;
      return profile(&callable);
    }
    int spinlock() {
      SpinLockCallable callable;
      return profile(&callable);
    }
    int tsi() {
      TSICallable callable;
      return profile(&callable);
    }
    int add() {
      AddCallable callable;
      return profile(&callable);
    }
    int cas() {
      CasCallable callable;
      return profile(&callable);
    }
    int memcpy(int64_t block_size, int64_t n_block) {
      MemcpyCallable callable(block_size, n_block);
      return profile(&callable);
    }
    int fastspinlock() {
      FastSpinLockCallable callable;
      return profile(&callable);
    }
    int cas128() {
      Cas128Callable callable;
      return profile(&callable);
    }
    int getus() {
      GetUsCallable callable;
      return profile(&callable);
    }
};
#include "cmd_args_parser.h"
#define report_error(err, ...) if (0 != err)fprintf(stderr, __VA_ARGS__);
const char* _usages = "Usages:\n"
  "\t# Common Environment nthread=1 time_limit=1\n"
  "\t%1$s mutex\n"
  "\t%1$s spinlock\n"
  "\t%1$s tsi\n"
  "\t%1$s add\n"
  "\t%1$s cas\n"
  "\t%1$s memcpy block_size n_block\n"
  "\t%1$s fastspinlock\n"
  "\t%1$s cas128\n"
  "\t%1$s getus\n";

int main(int argc, char** argv)
{
  int err = 0;
  Perf perf;
  setlocale(LC_ALL, "");
  if (-EAGAIN != (err = CmdCall(argc, argv, perf.mutex):-EAGAIN))
  {
    report_error(err, "mutex()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.spinlock):-EAGAIN))
  {
    report_error(err, "spinlock()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.tsi):-EAGAIN))
  {
    report_error(err, "tsi()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.add):-EAGAIN))
  {
    report_error(err, "add()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.cas):-EAGAIN))
  {
    report_error(err, "cas()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.memcpy, IntArg("block_size", "512"), IntArg("n_block", "1024")):-EAGAIN))
  {
    report_error(err, "memcpy()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.fastspinlock):-EAGAIN))
  {
    report_error(err, "fastspinlock()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.cas128):-EAGAIN))
  {
    report_error(err, "cas128()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.getus):-EAGAIN))
  {
    report_error(err, "getus()=>%d", err);
  }
  else
  {
    fprintf(stderr, _usages, argv[0]);
  }
  return err;
}
