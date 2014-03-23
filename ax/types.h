#ifndef __OB_AX_TYPES_H__
#define __OB_AX_TYPES_H__
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include "a0.h"
#include "printer.h"

// basic struct define
struct Buffer
{
  Buffer(): limit_(0), used_(0), buf_(NULL) {}
  ~Buffer() {}
  int parse(const char* spec) {
    int err = AX_SUCCESS;
    limit_ = used_ = strlen(spec);
    buf_ = (char*)spec;
    return err;
  }

  void dump() {
    fprintf(stderr, "dump buffer: limit=%ld used=%ld", limit_, used_);
  }
  uint64_t limit_;
  uint64_t used_;
  char* buf_;
};

#include "crc.h"
struct Record
{
  Record(): checksum_(0), magic_(0), len_(0), id_(0) {}
  ~Record() {}
  void set(uint32_t magic, uint32_t len, char* buf, uint64_t id) {
    magic_ = magic;
    len_ = len;
    id_ = id;
    memcpy(buf_, buf, len);
  }
  uint32_t get_record_len() const { return len_ + sizeof(*this); }
  uint64_t do_calc_checksum() const { return crc64_sse42(magic_, (const char*)&magic_, (int64_t)get_record_len());}
  void calc_checksum() { checksum_ = do_calc_checksum();}
  bool check_checksum(uint32_t magic) const { return magic_ == magic && checksum_ == do_calc_checksum(); }
  uint64_t checksum_;
  uint32_t magic_;
  uint32_t len_;
  uint64_t id_;
  char buf_[0];
};

struct Server
{
  Server(): ip_(0), port_(0) {}
  ~Server() {}
  void reset() { ip_ = 0; port_ = 0; }
  bool is_valid() const { return port_ > 0; }
  uint64_t get_id() const { return *(uint64_t*)this; }
  const char* repr(Printer& printer) const {
    const unsigned char* ip = (const unsigned char*)&ip_;
    return printer.new_str("%d.%d.%d.%d:%d", ip[0], ip[1], ip[2], ip[3], port_);
  }
  int parse(const char* spec) {
    int err = AX_SUCCESS;
    char* p = NULL;
    char ip[64] = "";
    if (NULL == spec)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL == (p = strchr(const_cast<char*> (spec), ':')))
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (p - spec + 1 > (int64_t)sizeof(ip))
    {
      err = AX_INVALID_ARGUMENT;
    }
    else
    {
      strncpy(ip, spec, min(p - spec, (int64_t)sizeof(ip)));
      ip[p - spec] = 0;
      ip_ = inet_addr(ip);
      port_ = atoi(p+1);
    }
    return err;
  }
  uint32_t ip_;
  uint32_t port_;
};

#endif /* __OB_AX_TYPES_H__ */
