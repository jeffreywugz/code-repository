#include <errno.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include "ax_common.h"

#define FAA(x, i) __sync_fetch_and_add((x), (i))
#define CAS(x, ov, nv) __sync_compare_and_swap((x), (ov), (nv))
#define AL(x) __atomic_load((x))
#define AS(x, v) __atomic_store((x), (v))

#define futex(...) syscall(SYS_futex,__VA_ARGS__)

inline int futex_wake(volatile int* p, int val)
{
  int err = 0;
  if (0 != futex((int*)p, FUTEX_WAKE_PRIVATE, val, NULL, NULL, 0))
  {
    err = errno;
  }
  return err;
}

inline int futex_wait(volatile int* p, int val, const timespec* timeout)
{
  int err = 0;
  if (0 != futex((int*)p, FUTEX_WAIT_PRIVATE, val, timeout, NULL, 0))
  {
    err = errno;
  }
  return err;
}

volatile int64_t __next_tid __attribute__((weak));
inline int64_t itid()
{
  static __thread int64_t tid = -1;
  return tid < 0? (tid = __sync_fetch_and_add(&__next_tid, 1)): tid;
}

struct TlValue
{
public:
  TlValue() { memset(items_, 0, sizeof(items_)); }
  ~TlValue() {}
  int64_t& get(){ return *(int64_t*)(items_ + itid()); }
private:
  char items_[AX_MAX_THREAD_NUM][CACHE_ALIGN_SIZE];
};

struct SpinLock
{
  struct Guard
  {
    Guard(SpinLock& lock): lock_(lock) {
      lock_.lock();
    }
    ~Guard() {
      lock_.unlock();
    }
    SpinLock& lock_;
  }
  
  bool try_lock() {
    return CAS(&lock_, 0, 1);
  }

  bool lock() {
    while(!try_lock())
    {
      PAUSE();
    }
  }
  bool unlock() {
    return CAS(&lock_, 1, 0);
  }
  uint64_t lock_;
};
