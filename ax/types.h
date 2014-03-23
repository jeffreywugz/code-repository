#ifndef __OB_AX_TYPES_H__
#define __OB_AX_TYPES_H__

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>

#define MAX_SERVER_COUNT 8
struct ServerList
{
  ServerList(): count_(0) {}
  ~ServerList() {}
  int parse(const char* spec) {
    int err = AX_SUCCESS;
    char* str = cstrdup(__alloca__, spec);
    StrTok tokenizer(str, ",");
    for(char* tok = NULL; AX_SUCCESS == err && NULL != (tok = tokenizer.next()); )
    {
      if (count_ >= MAX_SERVER_COUNT)
      {
        err = AX_SIZE_OVERFLOW;
      }
      else
      {
        err = servers_[count_++].parse(tok);
      }
    }
    return err;
  }
  int count_;
  Server servers_[MAX_SERVER_COUNT];
};

typedef uint64_t Term;
typedef uint64_t Pos;
struct Cursor
{
  Cursor(): term_(), pos_() {}
  ~Cursor() {}
  int parse(const char* spec) {
    int err = AX_SUCCESS;
    if (2 != scanf(spec, "%lu:%lu", &term_, &pos_))
    {
      err = AX_INVALID_ARGUMENT;
    }
    return err;
  }
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
  Cursor cursor_;
  TokenList group0_;
  TokenList group1_;
};

struct Ballot
{
  Term term_;
  Token self_;
  Token leader_;
};

struct LogEntry
{
  RecordHeader header_;
  Term term_;
  Pos pos_;
};
#endif /* __OB_AX_TYPES_H__ */
