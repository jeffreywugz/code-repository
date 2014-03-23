#ifndef __OB_AX_PRINTER_H__
#define __OB_AX_PRINTER_H__
#include "a0.h"

class Printer
{
public:
  Printer(int64_t buf_size = 4096): buf_(NULL), limit_(0), pos_(0) {
    char* buf = (char*)ax_alloc(buf_size);
    if (NULL != buf)
    {
      buf_ = buf;
      limit_ = buf_size;
      pos_ = 0;
    }
  }

  ~Printer() {
    if (NULL != buf_)
    {
      ax_free(buf_);
      buf_ = NULL;
    }
  }
  void reset() {
    pos_ = 0;
    if (NULL != buf_)
    {
      *buf_ = 0;
    }
  }
  char* get_str(){
    return NULL != buf_ && limit_ > 0? buf_: NULL;
  }
  char* append(const char* format, ...) {
    char* src = NULL;
    int64_t count = 0;
    va_list ap;
    va_start(ap, format);
    if (NULL != buf_ && limit_ > 0 && pos_ < limit_
        && pos_ + (count = vsnprintf(buf_ + pos_, limit_ - pos_, format, ap)) < limit_)
    {
      src = buf_ + pos_;
      pos_ += count;
    }
    va_end(ap);
    return src;
  }
  char* new_str(const char* format, ...) {
    char* src = NULL;
    int64_t count = 0;
    va_list ap;
    va_start(ap, format);
    if (NULL != buf_ && limit_ > 0 && pos_ < limit_
        && pos_ + (count = vsnprintf(buf_ + pos_, limit_ - pos_, format, ap)) + 1 < limit_)
    {
      src = buf_ + pos_;
      pos_ += count + 1;
    }
    va_end(ap);
    return src;
  }
private:
  char* buf_;
  int64_t limit_;
  int64_t pos_;
};

inline Printer& get_tl_printer()
{
  static Printer printer[AX_MAX_THREAD_NUM];
  return printer[itid()];
}

template<typename T>
const char* repr(T& t)
{
  return t.repr(get_tl_printer());
}

#endif /* __OB_AX_PRINTER_H__ */
