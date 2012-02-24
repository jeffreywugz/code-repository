#include <errno.h>
#include <pthread.h>
#include <sys/time.h>
#define EINIT 1001
#define DECLARE_COPY_AND_ASSIGN(TypeName)       \
  TypeName(const TypeName&);                    \
  void operator=(const TypeName&);

T max(T x, T y)
{
  return x > y? x: y;
}

T min(T x, T y)
{
  return x > y? y: x;
}

char* sf(char* buf, const int64_t len, int64_t& pos, const char* format, va_list ap)
{
  char* result = NULL;
  int64_t str_len = 0;
  if (NULL == buf || 0 > len || 0 > pos || pos > len)
  {}
  else if (0 > (str_len = vsnprintf(buf + pos, len - pos, format, ap)))
  {}
  else
  {
    result = buf + pos;
    pos += str_len + 1;
  }
  return result;
}

char* sf(char* buf, const int64_t len, int64_t& pos, const char* format, ...)
{
  va_list ap;
  char* result = NULL;
  va_start(ap, format);
  result = sf(cbuf, format, ap);
  va_end(ap);
  return result;
}

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

#define profile(expr) ({                                                \
      int64_t old_us = 0;                                               \
      int64_t new_us = 0;                                               \
      int64_t result = 0;                                               \
      old_us = get_usec();                                              \
      result = expr;                                                    \
      new_us = get_usec();                                              \
      printf("%s=>%ld in %ldms\n", #expr, result, new_us - old_us);     \
      new_us - old_us; })

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

struct Lock
{
  int64_t lock_;
  Lock(): lock_(0) {}
  ~Lock(){}
  bool try_lock()
  {
    return __sync_bool_compare_and_swap(&lock_, 0, 1);
  }
  bool unlock()
  {
    return __sync_bool_compare_and_swap(&lock_, 1, 0);
  }
};

struct LockGuard
{
  bool locked_;
  Lock* lock_;
  LockGuard(): locked_(false), lock_(NULL) {}
  ~LockGuard() {
    if (locked_ && NULL != lock_) {
      locked_ = false;
      lock_->unlock();
    }
  }
  bool try_lock(Lock& lock) {
    if (lock.try_lock())
    {
      lock_ = lock;
      locked_ = true;
    }
    return locked_;
  }
  private:
  DECLARE_COPY_AND_ASSIGN
};

class Stack
{
  private:
    DECLARE_COPY_AND_ASSIGN(Stack);
    Lock lock_;
    int64_t capacity_;
    int64_t top_;
    void** base_;
  public:
    Stack(): lock_(), capacity_(0), top_(0), base_(NULL) {}
    ~Stack() {
      if (NULL != base_)
      {
        delete []base_;
        base_ = NULL;
      }
    }
    bool is_inited() const {
      return NULL != base_;
    }

    int init(const int64_t capacity, void** base) {
      int err = 0;
      if (is_inited())
      {
        err = EINIT;
      }
      else if (0 > capacity || NULL == base)
      {
        err = EINVAL;
      }
      else
      {
        capacity_ = capacity;
        base_ = base;
      }
      return err;
    }

    int push(void* item) {
      int err = 0;
      LockGuard lock_guard;
      if (NULL == base_)
      {
        err = EINIT;
      }
      else if (!lock_guard.try_lock(lock_))
      {
        err = EAGAIN;
      }
      else if (top_ >= limit_)
      {
        err = ENOBUFS;
      }
      else
      {
        base_[top_++] = item;
      }
      return err;
    }

    int pop(void*& item) {
      int err = 0;
      LockGuard lock_guard;
      if (NULL == base_)
      {
        err = EINIT;
      }
      else if (!lock_guard.try_lock(lock_))
      {
        err = EAGAIN;
      }
      else if (top_ <= 0)
      {
        err = ENODATA;
      }
      else
      {
        item = base_[--top_];
      }
      return err;
    }
};

class Queue
{
  private:
    DECLARE_COPY_AND_ASSIGN(Queue);
    Lock lock_;
    int64_t capacity_;
    int64_t front_;
    int64_t rear_;
    void** base_;
  public:
    Queue(): lock_(), capacity_(0), front_(0), rear_(0), base_(NULL) {}
    ~Queue() {
      if (NULL != base_)
      {
        delete []base_;
        base_ = NULL;
      }
    }
    bool is_inited() const {
      return NULL != base_;
    }
    int init(const int64_t capacity, void** base) {
      int err = 0;
      if (is_inited())
      {
        err = EINIT;
      }
      else if (0 > capacity || NULL == base)
      {
        err = EINVAL;
      }
      else
      {
        capacity_ = capacity;
        base_ = base;
      }
      return err;
    }

    int64_t size() const {
      return NULL == base_? -1 :(rear_ + capacity_  - front_)% capacity_;
    }

    int push(void* item) {
      int err = 0;
      LockGuard lock_guard;
      if (NULL == base_)
      {
        err = EINIT;
      }
      else if (!lock_guard.try_lock(lock_))
      {
        err = EAGAIN;
      }
      else if (capacity_ - 1 == size())
      {
        err = ENOBUFS;
      }
      else
      {
        base_[rear_++] = item;
        rear_ %= capacity_;
      }
      return err;
    }

    int pop(void*& item) {
      int err = 0;
      LockGuard lock_guard;
      if (NULL == base_)
      {
        err = EINIT;
      }
      else if (!lock_guard.try_lock(lock_))
      {
        err = EAGAIN;
      }
      else if (0 == size())
      {
        err = ENODATA;
      }
      else
      {
        item = base_[++front_];
        front_ %= capacity_;
      }
      return err;
    }
};

struct BufHolder
{
  int64_t len_;
  char* buf_;
  BufHolder(): len_(0), buf_(NULL) {}
  ~BufHolder() {
    if (NULL != buf_)
    {
      free(buf_);
      buf_ = NULL;
      len_ = 0;
    }
  }
  char* get_buf(const int64_t len) {
    char* buf = NULL;
    if (NULL != buf_)
    {}
    else if (0 > len)
    {}
    else if (NULL == (buf = malloc(len)))
    {}
    else
    {
      buf_ = buf;
      len_ = len;
    }
    return buf;
  }
};

