#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "func.h"
#include "str.h"

int sh(const char* cmd)
{
  return NULL == cmd ? EINVAL: system(cmd);
}

char* popen(char* buf, const int64_t len, int64_t& pos, const char* cmd)
{
  char* result = NULL;
  FILE* fp = NULL;
  int64_t count = 0;
  if (NULL == buf || 0 > len || 0 > pos || pos > len)
  {}
  else if (NULL == (fp = popen(cmd, "r")))
  {}
  else
  {
    result = buf + pos;
    count = fread(buf + pos, 1, len - pos - 1, fp);
    buf[pos += count] = 0;
  }
  return result;
}

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

int read_file(const char* path, char* buf, const int64_t buf_size, int64_t& read_bytes)
{
  int err = 0;
  int fd = -1;
  if (NULL == path || NULL == buf || 0 >= buf_size)
  {
    err = EINVAL;
  }
  else if (0 > (fd = open(path, O_RDONLY)))
  {
    err = EIO;
  }
  else if (0 > (read_bytes = read(fd, buf, buf_size-1)))
  {
    err = EIO;
  }
  else
  {
    buf[read_bytes] = 0;
  }
  close(fd);
  return err;
}

int read_line(char* buf, const int64_t buf_size, int64_t& read_bytes, const char* prompt)
{
  int err = 0;
  if (NULL == buf || 0 >= buf_size || NULL == prompt)
  {
    err = EINVAL;
  }
  else if (0 > fprintf(stdout, "%s", prompt))
  {
    err = ESTR;
  }
  else if (NULL == fgets(buf, buf_size, stdin))
  {
    err = EEND;
  }
  return err;
}

class Interpreter
{
  public:
    Interpreter(){}
    virtual ~Interpreter(){}
    virtual int execute(const char* cmd) = 0;
};

int execute_str(Interpreter* interpreter, const char* str, bool keep_going, const char* delim)
{
  int err = 0;
  int execute_err = 0;
  char* tmp_str = NULL;
  char* statmt = NULL;
  if (NULL == interpreter || NULL == str)
  {
    err = EINVAL;
    log(ERROR, "execute_str(interpreter=%p, str='%s')=>%d", interpreter, str, err);
  }
  else if (NULL == (tmp_str = strdup(str)))
  {
    err = ENOMEM;
    log(ERROR, "strdup()=>%d", err);
  }
  while(0 == err)
  {
    if (NULL == (statmt = mystrsep(&tmp_str, delim)))
    {
      break;
    }
    else if (0 != (execute_err = interpreter->execute(statmt)))
    {
      if (!keep_going)
      {
        err = execute_err;
      }
      if (0 != execute_err)
      {
        log(ERROR, "interpreter->execute(statmt='%s')=>%d[may be syntax error]", statmt, execute_err);
      }
    }
  }
  if (NULL != tmp_str)
  {
    free(tmp_str);
  }
  return err;
}

int execute_file(Interpreter* interpreter, const char* path, bool keep_going, const char* delim)
{
  int err = 0;
  int64_t read_bytes  = 0;
  static char buf[MAX_STR_BUF_SIZE];
  if (NULL == interpreter || NULL == path || NULL == delim)
  {
    err = EINVAL;
    log(ERROR, "execute_str(interpreter=%p, path='%s', delim='%s')=>%d", interpreter, path, delim, err);
  }
  else if (0 != (err = read_file(path, buf, sizeof(buf), read_bytes)))
  {
    log(ERROR, "read_file(path='%s'), buf_size=%ld)=>%d", path, sizeof(buf), err);
  }
  else if (0 != (err = execute_str(interpreter, buf, keep_going, delim)))
  {
    log(ERROR, "execute_str(str='%s')=>%d", buf, err);
  }
  return err;
}

int execute_interactive(Interpreter* interpreter, bool keep_going, const char* delim, const char* prompt)
{
  int err = 0;
  int execute_err = 0;
  int64_t read_bytes = 0;
  static char buf[MAX_STR_BUF_SIZE];
  if (NULL == interpreter || NULL == delim || NULL == prompt)
  {
    err = EINVAL;
  }
  while(0 == err)
  {
    if (0 != (err = read_line(buf, sizeof(buf), read_bytes, prompt)))
    {
      if (EEND != err)
      {
        log(ERROR, "read_line(buf_size=%ld)=>%d", sizeof(buf), err);
      }
      else
      {
        log(INFO, "quit.");
        err = 0;
        break;
      }
    }
    else if (0 != (execute_err = execute_str(interpreter, buf, keep_going, delim)))
    {
      if (!keep_going)
      {
        err = execute_err;
      }
      log(ERROR, "execute_str(str='%s')=>%d", buf, err);
    }
  }
  return err;
}

#ifdef __TEST_UTILS_SH__
class TestInterpreter: public Interpreter
{
  public:
    TestInterpreter() {}
    virtual ~TestInterpreter() {}
    virtual int execute(const char* st) {
      printf("execute: %s\n", st);
      return 0;
    }
};    
int main(int argc, char *argv[])
{
  TestInterpreter interp;
  return execute_interactive(&interp, true, ";", "test> ");
}
#endif
