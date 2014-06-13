#include <locale.h>
#include "profile.h"

void busy()
{
  for(int64_t i = 0; i < 0; i++)
  {
    PAUSE();
  }
}
class MallocCallable
{
public:
  MallocCallable() {}
  ~MallocCallable() {}
  int do_once(int64_t idx) {
    UNUSED(idx);
    char* buf = (char*)malloc(1024);
    free(buf);
    return 0;
  }
private:
};

class MutexCallable
{
public:
  MutexCallable() { pthread_mutex_init(&mutex_, NULL); }
  ~MutexCallable() { pthread_mutex_destroy(&mutex_); }
  int do_once(int64_t idx) {
    UNUSED(idx);
    pthread_mutex_lock(&mutex_);
    busy();
    pthread_mutex_unlock(&mutex_);
    return 0;
  }
private:
  pthread_mutex_t mutex_ CACHE_ALIGNED;
};

class SpinLockCallable
{
public:
  SpinLockCallable() { pthread_spin_init(&spinlock_, 0); }
  ~SpinLockCallable() { pthread_spin_destroy(&spinlock_); }
  int do_once(int64_t idx) {
    UNUSED(idx);
    pthread_spin_lock(&spinlock_);
    busy();
    pthread_spin_unlock(&spinlock_);
    return 0;
  }
private:
  pthread_spinlock_t spinlock_ CACHE_ALIGNED;
};

class TSICallable
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

class AddCallable
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

class CasCallable
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

class SwapCallable
{
public:
  SwapCallable() { value_ = 0; }
  ~SwapCallable() {}
  int do_once(int64_t idx) {
    int64_t value = 1;
    __sync_lock_test_and_set(&value_, value, value + 1);
    return 0;
  }
private:
  volatile int64_t value_ CACHE_ALIGNED;
};

class MemcpyCallable
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

class GetUsCallable
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

class RDLockCallable
{
public:
  RDLockCallable(){
    pthread_rwlock_init(&rwlock_, NULL);
  }
  ~RDLockCallable(){
    pthread_rwlock_destroy(&rwlock_);
  }
public:
  int do_once(int64_t idx) {
    pthread_rwlock_rdlock(&rwlock_);
    pthread_rwlock_unlock(&rwlock_);
    return 0;
  }
private:
  pthread_rwlock_t rwlock_;
};

#include "fast-spinlock.h"
class FastSpinLockCallable
{
public:
  FastSpinLockCallable() { }
  ~FastSpinLockCallable() {}
  int do_once(int64_t idx) {
    spinlock_.lock();
    busy();
    spinlock_.unlock();
    return 0;
  }
private:
  FastSpinLock spinlock_;
};

#include "seq-lock.h"
class SeqLockCallable
{
public:
  SeqLockCallable() { }
  ~SeqLockCallable() {}
  int do_once(int64_t idx) {
    int64_t seq = spinlock_.lock();
    busy();
    spinlock_.unlock(seq);
    return 0;
  }
private:
  SeqLock spinlock_;
};

#include "mcs-lock.h"
class MCSLockCallable
{
public:
  MCSLockCallable() { }
  ~MCSLockCallable() {}
  int do_once(int64_t idx) {
    MCSLock::Node node;
    spinlock_.lock(&node);
    busy();
    spinlock_.unlock(&node);
    return 0;
  }
private:
  MCSLock spinlock_;
};

#include "clh-lock.h"
class CLHLockCallable
{
public:
  CLHLockCallable() { }
  ~CLHLockCallable() {}
  int do_once(int64_t idx) {
    spinlock_.lock();
    busy();
    spinlock_.unlock();
    return 0;
  }
private:
  CLHLock spinlock_;
};

#include "spin-queue.h"
class SpinQueueCallable
{
public:
  enum { QLEN = 1<<14 };
public:
  SpinQueueCallable() { queue_.init(QLEN, buf_); }
  virtual ~SpinQueueCallable() {} 
  inline int do_once(int64_t idx) {
    int err = 0;
    void* p = (void*)1;
    if (0 == (idx & 1))
    {
      err = queue_.push(p);
    }
    else
    {
      err = queue_.pop(p);
    }
    return err;
  }
private:
  SpinQueue queue_;
  void* buf_[QLEN];
};

#include "link-queue.h"
class LinkQueueCallable
{
public:
  typedef LinkQueue::Item Item;
  LinkQueueCallable() {
    Item* first = new Item();
    queue_.init(first);
  }
  virtual ~LinkQueueCallable() {} 
  inline int do_once(int64_t idx) {
    int err = 0;
    if (0 == (idx & 1))
    {
      Item* p = new Item();
      if (0 == (err = queue_.push(p)))
      {
        p = NULL;
      }
    }
    else
    {
      Item* ret = NULL;
      Item* p = NULL;
      if (0 == (err = queue_.pop(ret, p)))
      {
        delete p;
      }
    }
    return err;
  }
private:
  LinkQueue queue_;
};

#include "cas128.h"
class Cas128Callable
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
#define profile(X) do_profile<OneLoopCallable<X ## Callable> >()
class Perf
{
public:
  template<typename X>
  int do_profile()
  {
    X x;
    return do_profile(&x);
  }
  int do_profile(Callable* callable) {
    int64_t n_thread = _cfgi("n_thread", "1");
    int64_t time_limit = _cfgi("time_limit", "1") * 1000000;
    BaseWorker worker;
    return worker.set_thread_num(n_thread).par_do(callable, time_limit);
  }
  int malloc() { return profile(Malloc); }
  int mutex() { return profile(Mutex); }
  int spinlock() { return profile(SpinLock); }
  int tsi() { return profile(TSI); }
  int rdlock() { return profile(RDLock); }
  int add() { return profile(Add); }
  int cas() { return profile(Cas); }
  int swap() { return profile(Swap);}
  int memcpy(int64_t block_size, int64_t n_block){ return 0; }
  int fastspinlock() { return profile(FastSpinLock); }
  int seqlock() { return profile(SeqLock);}
  int mcslock() { return profile(MCSLock);}
  int clhlock() { return profile(CLHLock); }
  int spinqueue() { return profile(SpinQueue); }
  int linkqueue() { return profile(LinkQueue); }
  int cas128() { return profile(Cas128); }
  int getus() { return profile(GetUs);}
};
#include "cmd_args_parser.h"
#define report_error(err, ...) if (0 != err)fprintf(stderr, __VA_ARGS__);
const char* _usages = "Usages:\n"
  "\t# Common Environment nthread=1 time_limit=1\n"
  "\t%1$s malloc\n"
  "\t%1$s mutex\n"
  "\t%1$s rdlock\n"
  "\t%1$s spinlock\n"
  "\t%1$s tsi\n"
  "\t%1$s add\n"
  "\t%1$s cas\n"
  "\t%1$s swap\n"
  "\t%1$s memcpy block_size n_block\n"
  "\t%1$s fastspinlock\n"
  "\t%1$s seqlock\n"
  "\t%1$s mcslock\n"
  "\t%1$s clhlock\n"
  "\t%1$s spinqueue\n"
  "\t%1$s linkqueue\n"
  "\t%1$s cas128\n"
  "\t%1$s getus\n";

int main(int argc, char** argv)
{
  int err = 0;
  Perf perf;
  setlocale(LC_ALL, "");
  if (-EAGAIN != (err = CmdCall(argc, argv, perf.malloc):-EAGAIN))
  {
    report_error(err, "mutex()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.mutex):-EAGAIN))
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
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.rdlock):-EAGAIN))
  {
    report_error(err, "rdlock()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.add):-EAGAIN))
  {
    report_error(err, "add()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.cas):-EAGAIN))
  {
    report_error(err, "cas()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.swap):-EAGAIN))
  {
    report_error(err, "swap()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.memcpy, IntArg("block_size", "512"), IntArg("n_block", "1024")):-EAGAIN))
  {
    report_error(err, "memcpy()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.fastspinlock):-EAGAIN))
  {
    report_error(err, "fastspinlock()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.seqlock):-EAGAIN))
  {
    report_error(err, "seqlock()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.mcslock):-EAGAIN))
  {
    report_error(err, "mcslock()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.clhlock):-EAGAIN))
  {
    report_error(err, "clhlock()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.spinqueue):-EAGAIN))
  {
    report_error(err, "spinqueue()=>%d", err);
  }
  else if (-EAGAIN != (err = CmdCall(argc, argv, perf.linkqueue):-EAGAIN))
  {
    report_error(err, "linkqueue()=>%d", err);
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
