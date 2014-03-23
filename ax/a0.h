#ifdef ERRNO_DEF
ERRNO_DEF(SUCCESS, 0, "success")
ERRNO_DEF(FATAL_ERR, -1, "fatal")
ERRNO_DEF(INVALID_ARGUMENT, -2, "invalid argument")
ERRNO_DEF(CMD_ARGS_NOT_MATCH, -3, "cmd args not match")
ERRNO_DEF(NO_MEM, -4, "no memory")
ERRNO_DEF(INIT_TWICE, -5, "init twice")
ERRNO_DEF(NOT_INIT, -6, "not init")
ERRNO_DEF(EAGAIN, -7, "resource busy")
ERRNO_DEF(NOT_SUPPORT, -9, "not support")
ERRNO_DEF(NO_ENT, -10, "no entry")
ERRNO_DEF(IO_ERR, -11, "io error")
ERRNO_DEF(SIZE_OVERFLOW, -20, "array size overflow")
ERRNO_DEF(BUF_OVERFLOW, -21, "buf size overflow")
ERRNO_DEF(QUEUE_OVERFLOW, -22, "queue size overflow")
ERRNO_DEF(HASH_OVERFLOW, -23, "hash size overflow")
ERRNO_DEF(POOL_OVERFLOW, -24, "pool overflow")
ERRNO_DEF(STATE_NOT_MATCH, -27, "epoll ctl fail")
ERRNO_DEF(NOT_EXIST, -27, "entry not exist")
ERRNO_DEF(CALLBACK_NOT_SET, -28, "callback not set")
ERRNO_DEF(ID_NOT_MATCH, -29, "id_map id not match")
ERRNO_DEF(NIO_CAN_NOT_LOCK, -30, "nio can not lock")
ERRNO_DEF(PKT_CHECKSUM_ERR, -30, "nio can not lock")
ERRNO_DEF(DATA_ERR, -31, "data error")
ERRNO_DEF(SEQ_OVERFLOW, -32, "seq overflow")

ERRNO_DEF(EPOLL_CREATE_ERR, -1000, "epoll create fail")
ERRNO_DEF(EPOLL_WAIT_ERR, -1001, "epoll wait fail")
ERRNO_DEF(EPOLL_CTL_ERR, -1002, "epoll ctl fail")
ERRNO_DEF(SOCK_CREATE_ERR, -1010, "sock create fail")
ERRNO_DEF(SOCK_BIND_ERR, -1011, "sock bind fail")
ERRNO_DEF(SOCK_LISTEN_ERR, -1012, "sock listen fail")
ERRNO_DEF(SOCK_ACCEPT_ERR, -1013, "sock accept fail")
ERRNO_DEF(SOCK_READ_ERR, -1014, "sock read fail")
ERRNO_DEF(SOCK_WRITE_ERR, -1015, "sock write fail")
ERRNO_DEF(SOCK_HUP, -1016, "sock hup")
ERRNO_DEF(GET_SOCKOPT_ERR, -1011, "get sockopt fail")
ERRNO_DEF(EVENTFD_CREATE_ERR, -1020, "eventfd create fail")
ERRNO_DEF(EVENTFD_IO_ERR, -1021, "eventfd create fail")
ERRNO_DEF(PTHREAD_KEY_CREATE_ERR, -1040, "pthread_key_create fail")
ERRNO_DEF(FCNTL_ERR, -1100, "fcntl fail")
ERRNO_DEF(FLOCK_FAIL, -1101, "flock fail")

ERRNO_DEF(TERM_NOT_MATCH, -4000, "term not match")
ERRNO_DEF(TERM_STALE, -4001, "term stale")
ERRNO_DEF(CURSOR_LAGGED, -4002, "cursor lagged")
ERRNO_DEF(CURSOR_NOT_MATCH, -4003, "cursor lagged")
ERRNO_DEF(VOTE_OTHER, -4004, "vote other")
#endif

#ifdef PCODE_DEF
PCODE_DEF(RESPONSE, 0, "response")
PCODE_DEF(PING, 1, "ping")
PCODE_DEF(SET_LOG_LEVEL, 2, "set log level")
PCODE_DEF(SET_RUN_MODE, 3, "set run mode")
PCODE_DEF(INSPECT, 4, "inspect")
PCODE_DEF(CLOCK_GET, 4, "get clock")
#endif

#ifndef __OB_AX_A0_H__
#define __OB_AX_A0_H__

#define ERRNO_DEF(name, value, desc) const static int AX_ ## name = value; // desc
#include __FILE__
#undef ERRNO_DEF

#define PCODE_DEF(name, value, desc) const static int AX_ ## name = value; // desc
#include __FILE__
#undef PCODE_DEF

#define UNUSED(v) ((void)(v))
#define FAA(x, i) __sync_fetch_and_add((x), (i))
#define CAS(x, ov, nv) __sync_bool_compare_and_swap((x), (ov), (nv))
#define AL(x) __atomic_load_n((x), __ATOMIC_SEQ_CST)
#define AS(x, v) __atomic_store_n((x), (v), __ATOMIC_SEQ_CST)
#define MBR() __sync_synchronize()
#define PAUSE() asm("pause;\n")

#define AX_MAX_THREAD_NUM 1024
#define CACHE_ALIGN_SIZE 64
#define CACHE_ALIGNED __attribute__((aligned(CACHE_ALIGN_SIZE)))
#define WEAK_SYM __attribute__((weak))
#define DIO_ALIGN_SIZE 512
#define arrlen(x) (sizeof(x)/sizeof(x[0]))
#define strbool(x) ((x)?"true":"false")
#define DLOG(prefix, format, ...) if (__enable_dlog__) {int64_t cur_ts = get_us(); fprintf(stderr, "[%ld.%.6ld] %s %s:%d [%ld] " format "\n", cur_ts/1000000, cur_ts%1000000, #prefix, __FILE__, __LINE__, pthread_self(), ##__VA_ARGS__); }
bool __enable_dlog__ WEAK_SYM;

#define __STDC_LIMIT_MACROS
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <stdarg.h>
#include <pthread.h>
#include <sys/time.h>
#include <errno.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <new>

typedef uint64_t Id;
#define INVALID_ID (~0ul)

inline int64_t min(int64_t x, int64_t y)
{
  return ((x > y)? y: x);
}

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

inline bool is2n(uint64_t n)
{
  return 0 == (n & (n -1));
}

template<typename T>
int parse(T& t, const char* spec)
{
  return NULL == spec? AX_INVALID_ARGUMENT: t.parse(spec);
}

inline int64_t get_us()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

void* ax_alloc(size_t size, int mod_id=0);
void ax_free(void* p);

struct MemChunkCutter
{
  MemChunkCutter(int64_t limit, char* buf): limit_(limit), used_(0), buf_(buf) {}
  ~MemChunkCutter() {}
  char* alloc(int64_t size) {
    char* p = NULL;
    if (NULL == buf_)
    {}
    else if (used_ + size > limit_)
    {}
    else
    {
      p = buf_ + used_;
      used_ += size;
    }
    return p;
  }
  int64_t limit_;
  int64_t used_;
  char* buf_;
};

class MallocGuard
{
public:
  MallocGuard() {
    memset(allocated_, 0, sizeof(allocated_));
  }
  ~MallocGuard() {
    for(int i = 0; i < (int)arrlen(allocated_); i++)
    {
      if (allocated_[i])
      {
        ax_free(allocated_[i]);
      }
    }
  }
  char* alloc(int64_t size, int mod=0) {
    return (char*)ax_alloc(size, mod);
  }
private:
  char* allocated_[16];
};

template<typename T, typename Allocator>
int init_container(T& t, int64_t capacity, Allocator& allocator)
{
  return t.init(capacity, allocator.alloc(t.calc_mem_usage(capacity)));
}

#endif /* __OB_AX_A0_H__ */
