#include "profile.h"
#include <assert.h>
#include <string.h>

#define CACHE_ALIGN_SIZE 64
#define CACHE_ALIGNED __attribute__((aligned(CACHE_ALIGN_SIZE)))
#define PAUSE() asm("pause\n")

struct GetTSICallable: public Callable
{
  int64_t n_items_;
  pthread_key_t key_;
  GetTSICallable& set(int64_t n_items) {
    fprintf(stderr, "get_tsi_callable(n_items=%ld)\n", n_items);
    n_items_ = n_items;
    pthread_key_create(&key_, NULL);
    return *this;
  }
  int call(pthread_t thread, int64_t idx) {
    int err = 0;
    fprintf(stdout, "worker[%ld] run\n", idx);
    for(int64_t i = 0; i < n_items_; i++) {
      if (NULL == pthread_getspecific(key_))
      {
        pthread_setspecific(key_, (void*)(idx + 1));
      }
    }
    return err;
  }
};

int main(int argc, char** argv)
{
  int err = 0;
  BaseWorker worker;
  GetTSICallable callable CACHE_ALIGNED;
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
