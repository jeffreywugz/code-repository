#ifndef __OB_AX_AX_COMMON_H__
#define __OB_AX_AX_COMMON_H__

#include "a0.h"
#include "malloc.h"
#include "printer.h"
#include "mlog.h"
#define DLOG(prefix, format, ...) {get_tl_printer().reset(); get_mlog().append("[%ld] %s %s:%ld [%ld] " format "\n", get_us(), #prefix, __FILE__, __LINE__, pthread_self(), ##__VA_ARGS__); }


struct MemChunkCutter
{
  MemChunkCutter(int64_t limit, char* buf): limit_(limit), used_(0), buf_(buf) {}
  ~MemChunkCutter() {}
  int64_t limit_;
  int64_t used_;
  char* buf_;
};

template<typename T, typename Allocator>
int init_container(T& t, int64_t capacity, Allocator& allocator)
{
  return t.init(capacity, allocator.alloc(t.calc_mem_usage(capacity)));
}

#include "ax_types.h"
#endif /* __OB_AX_AX_COMMON_H__ */
