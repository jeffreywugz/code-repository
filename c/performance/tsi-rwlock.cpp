#include "profile.h"
#include <assert.h>
#include <string.h>

#define CACHE_ALIGN_SIZE 64
#define CACHE_ALIGNED __attribute__((aligned(CACHE_ALIGN_SIZE)))
#define PAUSE() asm("pause\n")

struct RWLock
{
  const static int64_t N_THREAD = 128;
  volatile int64_t read_ref_[N_THREAD][CACHE_ALIGN_SIZE/sizeof(int64_t)];
  volatile uint64_t thread_num_;
  volatile uint64_t write_uid_ CACHE_ALIGNED;
  pthread_key_t key_ CACHE_ALIGNED;
  RWLock()
  {
    write_uid_ = 0;
    memset((void*)read_ref_, 0, sizeof(read_ref_));
    pthread_key_create(&key_, NULL);
  }
  ~RWLock()
  {
    pthread_key_delete(key_);
  }
  void rdlock()
  {
    volatile int64_t* ref = (volatile int64_t*)pthread_getspecific(key_);
    if (NULL == ref)
    {
      pthread_setspecific(key_, (void*)(ref = read_ref_[__sync_fetch_and_add(&thread_num_, 1) % N_THREAD]));
    }
    while(true)
    {
      if (0 == write_uid_)
      {
        ref++;
        __sync_synchronize();
        if (0 == write_uid_)
        {
          break;
        }
        ref--;
      }
      PAUSE();
    }
  }
  void rdunlock()
  {
    int64_t* ref = (int64_t*)pthread_getspecific(key_);
    __sync_synchronize();
    (*ref)--;
  }
  void wrlock()
  {
    while(!__sync_bool_compare_and_swap(&write_uid_, 0, 1))
      ;
    for(uint64_t i = 0; i < thread_num_; i++)
    {
      while(*read_ref_[i] > 0)
        ;
    }
    __sync_synchronize();
  }
  void wrunlock()
  {
    __sync_synchronize();
    write_uid_ = 0;
  }
} CACHE_ALIGNED;
struct RWLockCallable: public Callable
{
  int64_t n_items_;
  RWLock lock_;
  RWLockCallable& set(int64_t n_items) {
    fprintf(stderr, "rwlock_callable(n_items=%ld)\n", n_items);
    n_items_ = n_items;
    return *this;
  }
  int call(pthread_t thread, int64_t idx) {
    int err = 0;
    fprintf(stdout, "worker[%ld] run\n", idx);
    for(int64_t i = 0; i < n_items_; i++) {
      lock_.rdlock();
      lock_.rdunlock();
    }
    return err;
  }
};

int main(int argc, char** argv)
{
  int err = 0;
  BaseWorker worker;
  RWLockCallable callable CACHE_ALIGNED;
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
