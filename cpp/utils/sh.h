#ifndef __UTILS_SH_UTILS_H__
#define __UTILS_SH_UTILS_H__

#include <stdlib.h>
#include "func.h"

int sh(const char* cmd);
char* popen(char* buf, const int64_t len, int64_t& pos, const char* cmd);
struct MgTool {
  const static int64_t DEFAULT_CBUF_LEN = 1<<12;
  CBuf cbuf;
  BufHolder buf_holder;
  MgTool(int64_t limit=DEFAULT_CBUF_LEN) {
    cbuf.set(buf_holder.get_buf(limit), limit);
  }
  ~MgTool() {
  }

  int reset() {
    int err = 0;
    cbuf.pos_ = 0;
    return err;
  }
    
  char* sf(const char* format, ...) {
    va_list ap;
    char* result = NULL;
    va_start(ap, format);
    result = ::sf(cbuf.buf_, cbuf.len_, cbuf.pos_, format, ap);
    va_end(ap);
    return result;
  }

  int sh(const char* format, ...) {
    va_list ap;
    int err = 0;
    va_start(ap, format);
    err = ::sh(::sf(cbuf.buf_, cbuf.len_, cbuf.pos_, format, ap));
    va_end(ap);
    return err;
  }

  char* popen(const char* format, ...) {
    va_list ap;
    char* result = NULL;
    va_start(ap, format);
    result = ::popen(cbuf.buf_, cbuf.len_, cbuf.pos_, ::sf(cbuf.buf_, cbuf.len_, cbuf.pos_, format, ap));
    va_end(ap);
    return result;
  }
};

int read_file(const char* path, char* buf, const int64_t buf_size, int64_t& read_bytes);
int read_line(char* buf, const int64_t buf_size, int64_t& read_bytes, const char* prompt);
class Interpreter
{
  public:
    Interpreter(){}
    virtual ~Interpreter(){}
    virtual int execute(const char* cmd) = 0;
};

int execute_str(Interpreter* interpreter, const char* str, bool keep_going, const char* delim);
int execute_file(Interpreter* interpreter, const char* path, bool keep_going, const char* delim);
int execute_interactive(Interpreter* interpreter, bool keep_going, const char* delim, const char* prompt);
#endif /* __UTILS_SH_UTILS_H__ */
