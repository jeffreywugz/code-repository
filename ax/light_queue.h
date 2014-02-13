#include "mcu.h"

class ObLightyQueue
{
public:
  struct Item
  {
    Item(int64_t seq): n_waiters_(0), stock_(0), data_(NULL) { UNUSED(seq); }
    ~Item() {}
    int push(void* data) {
      int err = OB_SUCCESS;
      if (NULL == data)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else
      {
        __sync_fetch_and_add(&stock_, 1);
        while(!__sync_bool_compare_and_swap(&data_, NULL, data))
        {
          PAUSE();
        }
        if (n_waiters_ > 0)
        {
          futex(&stock_, FUTEX_WAKE, INT32_MAX, NULL, NULL, 0);
        }
      }
      return err;
    }
    int pop(void*& data, const timespec* timeout) {
      int err = OB_SUCCESS;
      int32_t stock = 0;
      if ((stock = dec_if_gt0(&stock_)) > 0)
      {}
      else
      {
        __sync_add_and_fetch(&n_waiters_, 1);
        while(OB_SUCCESS == err)
        {
          if (ETIMEDOUT == futex_wait(&stock_, stock, timeout))
          {
            err = OB_EAGAIN;
          }
          else if ((stock = dec_if_gt0(&stock_)) > 0)
          {
            break;
          }
        }
        __sync_add_and_fetch(&n_waiters_, -1);
      }
      if (OB_SUCCESS == err)
      {
        while(NULL == (data = data_) || !__sync_bool_compare_and_swap(&data_, data, NULL))
        {
          PAUSE();
        }
      }
      return err;
    }
    volatile int32_t n_waiters_;
    volatile int32_t stock_;
    void* volatile data_;
  };
public:
  ObLightyQueue(): push_(1), pop_(1) {}
  ~ObLightyQueue() {}
  int init(int64_t qlen) { return items_.init(qlen); }
  int push(void* data) {
    int err = OB_SUCCESS;
    Item* item = NULL;
    if (NULL == data)
    {
      err = OB_INVALID_ARGUMENT;
    }
    else if (NULL == (item = items_.get(__sync_fetch_and_add(&push_, 1))))
    {
      err = OB_NOT_INIT;
    }
    else
    {
      err = item->push(data);
    }
    return err;
  }

  int pop(void*& data, const timespec* timeout) {
    int err = OB_SUCCESS;
    int64_t& idx = ticket_.get();
    Item* item = NULL;
    if (idx == 0)
    {
      idx = __sync_fetch_and_add(&pop_, 1);
    }
    if (NULL == (item = items_.get(idx)))
    {
      err = OB_NOT_INIT;
    }
    else if (OB_SUCCESS == (err = item->pop(data, timeout)))
    {
      idx = 0;
    }
    return err;
  }
private:
  volatile int64_t push_ CACHE_ALIGNED;
  volatile int64_t pop_ CACHE_ALIGNED;
  TCValue ticket_;
  ObFixedArray<Item> items_;
};
