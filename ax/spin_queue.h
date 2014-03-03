#ifndef __OB_AX_SPIN_QUEUE_H__
#define __OB_AX_SPIN_QUEUE_H__
#include "a0.h"

class SpinQueue
{
public:
  SpinQueue(): push_(0), pop_(0), capacity_(0), items_(NULL) {}
  ~SpinQueue(){ destroy(); }
  int init(int64_t capacity, char* buf) {
    int err = AX_SUCCESS;
    if (capacity < 0 || !is2n(capacity) || NULL == buf)
    {
      err = AX_NOT_INIT;
    }
    else
    {
      capacity_ = capacity;
      items_ = (void**)buf;
      memset(items_, 0, calc_mem_usage(capacity));
    }
    return err;
  }
  void destroy() {
    push_ = 0;
    pop_ = 0;
    capacity_ = 0;
    items_ = 0;
  }
  static int64_t calc_mem_usage(int64_t capacity) { return sizeof(void*) * capacity; }
  int64_t idx(int64_t x) { return x & (capacity_ - 1); }
  int push(void* p) {
    int err = AX_EAGAIN;
    int64_t push = -1;
    if (NULL == p)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL == items_)
    {
      err = AX_NOT_INIT;
    }
    else
    {
      while((push = AL(&push_)) < AL(&pop_) + capacity_)
      {
        if (!CAS(&push_, push, push + 1))
        {
          PAUSE();
        }
        else
        {
          while(!CAS(items_ + idx(push), NULL, p))
          {
            PAUSE();
          }
          err = AX_SUCCESS;
          break;
        }
      }
    }
    return err;
  }
  int pop(void*& p) {
    int err = AX_EAGAIN;
    int64_t pop = -1;
    if (NULL == items_)
    {
      err = AX_NOT_INIT;
    }
    else
    {
      while((pop = AL(&pop_)) < AL(&push_))
      {
        if (!CAS(&pop_, pop, pop + 1))
        {
          PAUSE();
        }
        else
        {
          void** pdata = items_ + idx(pop);
          while(NULL == (p = AL(pdata)) || !CAS(pdata, p, NULL))
          {
            PAUSE();
          }
          err = AX_SUCCESS;
          break;
        }
      }
    }
    return err;
  }
private:
  int64_t push_ CACHE_ALIGNED;
  int64_t pop_ CACHE_ALIGNED;
  int64_t capacity_;
  void** items_;
};
#endif /* __OB_AX_SPIN_QUEUE_H__ */
