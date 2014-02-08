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

// basic struct define
struct Buffer
{
  Buffer(): limit_(0), used_(0), buf_(NULL) {}
  ~Buffer() {}
  int parse(const char* spec);
  void dump();
  uint64_t limit_;
  uint64_t used_;
  char* buf_;
};

struct RecordHeader
{
  uint32_t magic_;
  uint32_t len_;
  uint64_t checksum_;
};

struct Server
{
  Server(): ip_(0), port_(0) {}
  ~Server() {}
  int parse(const char* spec);
  uint32_t ip_;
  uint32_t port_;
};

#define MAX_SERVER_COUNT 8
struct ServerList
{
  ServerList(): count_(0) {}
  ~ServerList() {}
  int parse(const char* spec);
  int count_;
  Server servers_[MAX_SERVER_COUNT];
};

typedef uint64_t Term;
typedef uint64_t Pos;
struct Cursor
{
  Cursor(): term_(), pos_() {}
  ~Cursor() {}
  int parse(const char* spec);
  Term term_;
  Pos pos_;
};

struct Token
{
  Server server_;
  uint64_t start_time_;
};

#define MAX_TOKEN_COUNT 8
struct TokenList
{
  TokenList(): count_(0) {}
  ~TokenList() {}
  int count_;
  Token tokens_[MAX_TOKEN_COUNT];
};

struct GroupConfig
{
  TokenList active_tokens_;
  ServerList group0_;
  ServerList group1_;
};

struct MetaCheckPoint
{
  Cursor frozen_cursor_;
  GroupConfig group_config_;
};

struct MetaRecord
{
  RecordHeader header_;
  uint64_t version_;
  Token self_;
  Term term_;
  Server leader_;
  Cursor commit_cursor_;
  GroupConfig group_config_;
};

struct LogEntry
{
  RecordHeader header_;
  Term term_;
  Pos pos_;
};

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
#include "debug_log.h"

#endif /* __OB_AX_AX_COMMON_H__ */
