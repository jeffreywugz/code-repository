#include "btree.h"

#define is_env_set(key, _default) (0 == (strcmp("true", getenv(key)?: _default)))
#define cfg(key, default_value) (getenv(key)?:default_value)
#define cfgi(key, default_value) atoi(cfg(key, default_value))

typedef void*(*pthread_handler_t)(void*);
int64_t rand2(int64_t h)
{
  if (0 == h) return 1;
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccd;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53;
  h ^= h >> 33;
  h &= ~(1ULL<<63);
  return h;
}

class TestBtree
{
public:
  TestBtree(): limit_(0), insert_count_(0), get_count_(0) {
  }
  ~TestBtree() {}
  static int64_t randint() {
    static int64_t g_idx = 0;
    static __thread int64_t i = 0;
    if (i == 0)
    {
      i = (FAA(&g_idx, 1) + 1) * (1LL<<32);
    }
    return rand2(i++);
  }
  int test(int64_t n_thread, int64_t limit) {
    pthread_t thread[1024];
    printf("test: n_thread=%ld limit=%ld\n", n_thread, limit);

    limit_ = limit;
    pthread_barrier_init(&wait_insert_, NULL, n_thread + 1);

    {
      int64_t start_ts = get_us();
      for(int64_t i = 0; i < n_thread; i++) {
        pthread_create(thread + i, NULL, (pthread_handler_t)thread_func, this);
      }
      pthread_barrier_wait(&wait_insert_);
      int64_t end_ts = get_us();
      printf("insert: %ld/%ld=%'ld\n", insert_count_ * 1000000, end_ts - start_ts, insert_count_ * 1000000/(end_ts - start_ts));
    }

    {
      int64_t start_ts = get_us();
      for(int64_t i = 0; i < n_thread; i++) {
        pthread_join(thread[i], NULL);
      }
      int64_t end_ts = get_us();
      printf("get: %ld/%ld=%'ld\n", get_count_ * 1000000, end_ts - start_ts, get_count_ * 1000000/(end_ts - start_ts));
    }

    if (is_env_set("dump_btree", "false"))
    {
      btree_.print();
    }
    return 0;
  }
  int do_work() {
    int err = BTREE_SUCCESS;
    const int64_t batch_limit = 64;
    int64_t insert_count = 0;
    while(0 == err)
    {
      if ((insert_count = FAA(&insert_count_, batch_limit)) + batch_limit >= limit_)
      {
        FAA(&insert_count_, -batch_limit);
        break;
      }
      for(int64_t i = 0; i < batch_limit; i++) {
        int64_t key = rand2(insert_count + i);
        while(BTREE_EAGAIN == (err = btree_.set(key, (void*)key)))
          ;
        if (0 != err)
        {
          BTREE_LOG(ERROR, "btree.set(%ld)=>%d", key, err);
          break;
        }
      }
    }

    int64_t get_count = 0;
    pthread_barrier_wait(&wait_insert_);
    while(0 == err)
    {
      if ((get_count = FAA(&get_count_, batch_limit)) + batch_limit >= limit_)
      {
        FAA(&get_count_, -batch_limit);
        break;
      }
      for(int64_t i = 0; i < batch_limit; i++) {
        int64_t key = rand2(get_count + i);
        int64_t val = 0;
        if (0 != (err = btree_.get(key, (void*&)val)))
        {
          BTREE_LOG(ERROR, "btree.get(%ld)=>%d seq=%ld", key, err, get_count + i);
          break;
        }
      }
    }
    return 0;
  }
  static void* thread_func(TestBtree* host) {
    host->do_work();
    return NULL;
  }
private:
  int64_t limit_;
  int64_t insert_count_;
  int64_t get_count_;
  Btree btree_;
  pthread_barrier_t wait_insert_;
  pthread_barrier_t wait_get_;
};

#include <locale.h>
int main(int argc, char** argv)
{
  int err = 0;
  TestBtree test_btree;
  int64_t n_thread = cfgi("n_thread", "1");
  int64_t total_count = cfgi("total_count", "1000000");
  __enable_dlog__ = true;
  setlocale(LC_ALL, "");
  test_btree.test(n_thread, total_count);
  return err;
}
