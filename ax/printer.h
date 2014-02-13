class Printer
{
public:
  Printer(const char* buf, int64_t len): buf_(buf), limit_(len), pos_(0) {
    if (NULL != buf_ && limit_ > 0)
    {
      *buf_ = 0;
    }
  }
  ~Printer() {}
  char* get_str(){
    return NULL != buf_ && limit_ > 0? buf_: NULL;
  }
  char* append(const char* format, ...) {
    char* src = NULL;
    int64_t count = 0;
    va_list ap;
    va_start(ap, format);
    if (NULL != buf_ && limit_ > 0 && pos_ < limit_
        && pos_ + (count = vsnprintf(buf_ + pos_, len_ - pos_, format, ap)) < len_)
    {
      src = buf_ + pos_
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
        && pos_ + (count = vsnprintf(buf_ + pos_, len_ - pos_, format, ap)) < len_)
    {
      src = buf_ + pos_
      pos_ += count;
    }
    va_end(ap);
    return src;
  }
private:
  char* buf_;
  int64_t limit_;
  int64_t pos_;
};

template<typename T>
int format
