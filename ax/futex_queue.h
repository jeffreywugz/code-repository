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
  ObLightyQueue(): push_(1), pop_(1) {}
  ~ObLightyQueue() {}
  int init(int64_t qlen) { return items_.init(qlen); }
  int push(void* data) {
    int err = AX_SUCCESS;
    Item* item = NULL;
    if (NULL == data)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL == (item = items_.get(FAA(&push_, 1))))
    {
      err = AX_NOT_INIT;
    }
    else
    {
      err = item->push(data);
    }
    return err;
  }

  int pop(void*& data, const timespec* timeout) {
    int err = AX_SUCCESS;
    int64_t& idx = ticket_.get();
    Item* item = NULL;
    if (idx == 0)
    {
      idx = FAA(&pop_, 1);
    }
    if (NULL == (item = items_.get(idx)))
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS == (err = item->pop(data, timeout)))
    {
      idx = 0;
    }
    return err;
  }
private:
  int64_t push_ CACHE_ALIGNED;
  int64_t pop_ CACHE_ALIGNED;
  TCValue ticket_;
  FixedArray<Item> items_;
};
