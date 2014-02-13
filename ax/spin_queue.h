#include "ax_common.h"
#include "ob_fixed_array.h"

class SpinQueue
{
public:
  struct Item
  {
    Item(int64_t seq): data_(NULL) { UNUSED(seq); }
    ~Item() {}
    void* data_;
  };
public:
  SpinQueue(): push_(0), pop_(0) {}
  ~SpinQueue(){}
  int init(int64_t size) { return items_.init(size); }
  int push(void* p) {
    int err = AX_EAGAIN;
    int64_t push = -1;
    if (NULL == p)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (!items_.is_inited())
    {
      err = AX_NOT_INIT;
    }
    else
    {
      while((push = AL(&push_)) < AL(&pop_) + items_.len())
      {
        if (!CAS(&push_, push, push + 1))
        {
          PAUSE();
        }
        else
        {
          while(!CAS(&items_.get(push)->data_, NULL, p))
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
    if (!items_.is_inited())
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
          Item* item = items_.get(pop);
          while(NULL == (p = AL(&item->data_))
                || !CAS(&item->data_, p, NULL))
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
  FixedArray<Item> items_;
};
