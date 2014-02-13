#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

class MLog
{
public:
  enum { BUF_SIZE = 1<<12 };
public:
  MLog(const char* path): fd_(-1) {
    if (NULL != path)
    {
      fd_ = open(path, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    }
    else
    {
      fd_ = 1;
    }
  }
  ~MLog() {
    // if (fd_ > 0)
    // {
    //   close(fd_);
    //   fd_ = -1;
    // }
  }
public:
  int append(const char* format, ...) {
    int err = AX_SUCCESS;
    char buf[BUF_SIZE];
    int64_t count = 0;
    int64_t write_size = 0;
    int64_t write_ret = 0;
    va_list ap;
    if (NULL == format)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (fd_ < 0)
    {
      err = AX_NOT_INIT;
    }
    else
    {
      va_start(ap, format);
      count = vsnprintf(buf, sizeof(buf), format, ap);
      va_end(ap);
    }
    if (AX_SUCCESS != err)
    {}
    else if (count < 0)
    {
      err = AX_FATAL_ERR;
    }
    else if (count > (int64_t)sizeof(buf))
    {
      count = sizeof(buf);
    }
    while(AX_SUCCESS == err && write_size < count)
    {
      if ((write_ret = write(fd_, buf + write_size, count - write_size)) > 0)
      {
        write_size += write_ret;
      }
      else if (EINTR == errno)
      {}
      else
      {
        err = AX_IO_ERR;
      }
    }
    return err;
  }
private:
  int fd_;
};
inline MLog& get_mlog() {
  static MLog mlog(getenv("mlog_path"));
  return mlog;
}
#define DLOG(prefix, format, ...) get_mlog().append("[%ld] %s %s:%ld [%ld] " format "\n", get_us(), #prefix, __FILE__, __LINE__, pthread_self(), ##__VA_ARGS__)
