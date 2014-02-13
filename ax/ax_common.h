#ifdef ERRNO_DEF
ERRNO_DEF(SUCCESS, 0, "success")
ERRNO_DEF(FATAL_ERR, -1, "fatal")
ERRNO_DEF(INVALID_ARGUMENT, -2, "invalid argument")
ERRNO_DEF(CMD_ARGS_NOT_MATCH, -3, "cmd args not match")
ERRNO_DEF(NO_MEM, -4, "no memory")
ERRNO_DEF(INIT_TWICE, -5, "init twice")
ERRNO_DEF(NOT_INIT, -6, "not init")
ERRNO_DEF(IO_ERR, -9, "io error")
ERRNO_DEF(SIZE_OVERFLOW, -19, "array size overflow")
ERRNO_DEF(BUF_OVERFLOW, -20, "buf size overflow")
#endif

#ifdef PCODE_DEF
PCODE_DEF(PING, 1, "ping")
PCODE_DEF(SET_LOG_LEVEL, 2, "set log level")
PCODE_DEF(SET_RUN_MODE, 3, "set run mode")
PCODE_DEF(INSPECT, 4, "inspect")
#endif

#ifndef __OB_AX_AX_COMMON_H__
#define __OB_AX_AX_COMMON_H__
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

#define ERRNO_DEF(name, value, desc) const static int AX_ ## name = value; // desc
#include __FILE__
#undef ERRNO_DEF

#define PCODE_DEF(name, value, desc) const static int AX_ ## name = value; // desc
#include __FILE__
#undef PCODE_DEF

#define UNUSED(v) ((void)(v))

// sys config
#define AX_MAX_THREAD_NUM 1024
#define CACHE_ALIGN_SIZE 64
#define CACHE_ALIGNED __attribute__((aligned(CACHE_ALIGN_SIZE)))
#define DIO_ALIGN_SIZE 512

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

void *(*ax_malloc)(size_t size) = malloc;
void (*ax_free)(void *ptr) = free;

#include "ax_types.h"
#include "debug_log.h"
#endif /* __OB_AX_AX_COMMON_H__ */
