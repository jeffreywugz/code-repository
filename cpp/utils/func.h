#ifndef __UTILS_FUNC_UTILS_H__
#define __UTILS_FUNC_UTILS_H__
#include <stdlib.h>
#include "common.h"

struct CBuf {
  CBuf(): buf_(NULL), len_(0), pos_(0) {}
  ~CBuf() {}
  char* buf_;
  int64_t len_;
  int64_t pos_;
  int set(char* buf, int64_t len) {
    buf_ = buf;
    len_ = len;
    pos_ = 0;
    return 0;
  }
};

struct BufHolder
{
  int64_t n_pointers_;
  char* pointers_[256];
  BufHolder(): n_pointers_(0) {}
  ~BufHolder() {
    for(int i = 0; i < n_pointers_; i++) {
      free(pointers_[i]);
      n_pointers_ = 0;
    }
  }
  char* get_buf(const int64_t len) {
    char* buf = NULL;
    if (0 > len)
    {}
    else if (NULL == (buf = (char*)malloc(len)))
    {}
    else
    {
      pointers_[n_pointers_++] = buf;
    }
    return buf;
  }
};

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
      lock_ = &lock;
      locked_ = true;
    }
    return locked_;
  }
  private:
    DECLARE_COPY_AND_ASSIGN(LockGuard);
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
      else if (top_ >= capacity_)
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

#endif /* __UTILS_FUNC_UTILS_H__ */
