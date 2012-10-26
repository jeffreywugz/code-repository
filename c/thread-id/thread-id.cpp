#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <assert.h>

//#define mfence() __asm__ volatile("mfence;\n");
#define mfence() __sync_synchronize()
#define mfence()
#define clear_bit(x, n) ((x) &= ~(1ULL<<(n)))
#define set_bit(x, n) ((x) |= (1ULL<<(n)))
#define first_1bit(x) ((int64_t)__builtin_ffsll(x) - 1)

class SmallIdSet
{
  public:
    // 1 indicate free, 0 indicate used
    static const int64_t unit_size = 64;
    static const uint64_t lock_mask = (1ULL<<(unit_size-1));
  public:
    SmallIdSet(): lock_(0) { clear(); pthread_mutex_init(&mutex_, NULL); }
    ~SmallIdSet() {}
    void clear() { header_ = ~0ULL; memset((void*)bits_, ~0u, sizeof(bits_)); }
#if 0
    bool lock() {
      pthread_mutex_lock(&mutex_);
    }
    bool unlock() {
      pthread_mutex_unlock(&mutex_);
    }
#else    
    bool lock() {
      bool locked = __sync_bool_compare_and_swap(&lock_, 0ULL, 1ULL);
      mfence();
      return locked;
    }
    bool unlock() {
      mfence();
      return __sync_bool_compare_and_swap(&lock_, 1ULL, 0ULL);
    }
#endif    
    int alloc(int64_t& id) {
      int err = 0;
      int64_t unit_idx = 0;
      bool locked = true;
      if (!lock())
      {
        locked = false;
        err = -EAGAIN;
      }
      else if (0 > (unit_idx = first_1bit(header_))
               || 0 > (id = first_1bit(bits_[unit_idx])))
      {
        err = -EAGAIN;
      }
      else
      {
        clear_bit(bits_[unit_idx], id);
        if (0 == bits_[unit_idx])
          clear_bit(header_, unit_idx);
        id += unit_idx * unit_size;
      }
      if (locked && !unlock())
      {
        err = -EDOM;
      }
      return err;
    }
    int free(const int64_t id) {
      int err = 0;
      bool locked = true;
      if (!lock())
      {
        locked = false;
        err = -EAGAIN;
      }
      else if (id < 0 || id >= unit_size * unit_size)
      {
        err = -EINVAL;
      }
      else
      {
        set_bit(bits_[id/unit_size], id%unit_size);
        set_bit(header_, id/unit_size);
      }
      if (locked && !unlock())
      {
        err = -EDOM;
      }
      return err;
    }
    int check() {
      uint64_t std_bits[unit_size];
      memset(std_bits, ~0U, sizeof(std_bits));
      return memcmp((void*)bits_, std_bits, sizeof(std_bits));
    }
  private:
    pthread_mutex_t mutex_;
    volatile int64_t lock_;
    uint64_t header_;
    uint64_t bits_[unit_size];
};

int64_t g_n = 0;
SmallIdSet id_set;
void* do_work(void* arg)
{
  int err = 0;
  int64_t id = 0;
  int64_t thread_idx = (int64_t)arg;
  for(int64_t i = 0; 0 == err && i < g_n; i++)
  {
    while(-EAGAIN == (err = id_set.alloc(id)))
      ;
    if (0 != err)
    {
      printf("error:%d\n", err);
      continue;
    }
    usleep(10000);
    while(-EAGAIN == (err = id_set.free(id)))
      ;
    printf("thread[%ld], get_id: %ld, err=%d\n", thread_idx, id, err);
  }
  printf("thread[%ld], exit err=%d\n", thread_idx, err);
}

typedef void*(*pthread_handler_t)(void*);
int test_id_set(int64_t n_thread, int64_t n)
{
  int err = 0;
  pthread_t thread[n_thread];
  g_n = n;
  for(int64_t i = 0; i < n_thread; i++)
    pthread_create(thread + i, NULL, (pthread_handler_t)(do_work), (void*)i);
  for(int64_t i = 0; i < n_thread; i++)
    pthread_join(thread[i], NULL);
  assert(id_set.check() == 0);
  return err;
}

int seq_test(int64_t n)
{
  int err = 0;
  int64_t id = 0;
  for(int64_t i = 0; i < n; i++)
  {
    assert(0 == id_set.alloc(id));
    assert(id == i);
  }
  for(int64_t i = 0; i < n; i++)
  {
    assert(0 == id_set.free(i));
  }
  assert(0 == id_set.check());
}

int main(int argc, char *argv[])
{
  int err = 0;
  if (argc != 3)
  {
    fprintf(stderr, "%s n_thread n_id\n", argv[0]);
    err = -EINVAL;
  }
  else
  {
    err = test_id_set(atoll(argv[1]), atoll(argv[2]));
  }
  return err;
}
