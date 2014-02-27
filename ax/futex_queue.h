#include "mcu.h"

class FutexQueue
{
public:
  struct Item
  {
    Item(int64_t seq): n_waiters_(0), stock_(0), data_(NULL) { UNUSED(seq); }
    ~Item() {}
    int push(void* data) {
      int err = AX_SUCCESS;
      if (NULL == data)
      {
        err = AX_INVALID_ARGUMENT;
      }
      else
      {
        FAA(&stock_, 1);
        while(!CAS(&data_, NULL, data))
        {
          PAUSE();
        }
        if (AL(&n_waiters_) > 0)
        {
          futex(&stock_, FUTEX_WAKE, INT32_MAX, NULL, NULL, 0);
        }
      }
      return err;
    }
    int pop(void*& data, const timespec* timeout) {
      int err = AX_SUCCESS;
      int32_t stock = 0;
      if ((stock = dec_if_gt0(&stock_)) > 0)
      {}
      else
      {
        FAA(&n_waiters_, 1);
        while(AX_SUCCESS == err)
        {
          if (ETIMEDOUT == futex_wait(&stock_, stock, timeout))
          {
            err = AX_EAGAIN;
          }
          else if ((stock = dec_if_gt0(&stock_)) > 0)
          {
            break;
          }
        }
        FAA(&n_waiters_, -1);
      }
      if (AX_SUCCESS == err)
      {
        while(NULL == (data = AL(&data_)) || !CAS(&data_, data, NULL))
        {
          PAUSE();
        }
      }
      return err;
    }
    int32_t n_waiters_;
    int32_t stock_;
    void* data_;
  };
public:
  FutexQueue(): push_(1), pop_(1), size_(0), items_(NULL) {}
  ~FutexQueue() { destroy(); }
  static int64_t calc_mem_usage(int64_t size) { return sizeof(Item) * size; }
  int init(int64_t len, void* data) {
    int err = AX_SUCCESS;
    if (len <= 0 || !is2n(len) || NULL == data)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (0 != pthread_key_create(&key_, NULL))
    {
      err = AX_PTHREAD_KEY_CREATE_ERR;
    }
    else
    {
      len_ = len;
      items_ = data;
      memset(data, 0, calc_mem_usage(len));
    }
    return err;
  }
  void destroy() {
    if (NULL != items_)
    {
      push_ = 1;
      pop_ = 1;
      size_ = 0;
      items_ = NULL;
      pthread_key_destroy(key_);
    }
  }
  int64_t idx(int64_t x) { return x & (len_ - 1); }
  int push(void* data) {
    int err = AX_SUCCESS;
    Item* item = NULL;
    if (NULL == data)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL == items_)
    {
      err = AX_NOT_INIT;
    }
    else
    {
      err = items_[idx(FAA(&push_, 1))].push(data);
    }
    return err;
  }

  int pop(void*& data, const timespec* timeout) {
    int err = AX_SUCCESS;
    int64_t cur_idx = pthread_getspecific(key_)?:FAA(&pop_, 1);
    if (NULL == items_)
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS != (err = items_[idx(cur_idx)].pop(data, timeout)))
    {
      pthread_setspecific(key_, cur_idx);
    }
    else
    {
      pthread_setspecific(key_, 0);
    }
    return err;
  }
private:
  int64_t push_ CACHE_ALIGNED;
  int64_t pop_ CACHE_ALIGNED;
  pthread_key_t key_;
  int64_t len_ CACHE_ALIGNED;
  Item** items_;
};
