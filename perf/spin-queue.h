
class FastSpinQueue
{
public:
  FastSpinQueue(): count_(0), push_(0), pop_(0), capacity_(0), items_(NULL) {}
  ~FastSpinQueue(){}
  int init(uint64_t capacity, void** items) {
    int err = 0;
    capacity_ = capacity;
    memset(items, 0, sizeof(void*) * capacity);
    items_ = items;
    return err;
  }
  int destroy() {
    int err = 0;
    items_ = NULL;
    return err;
  }
  int64_t idx(int64_t x) { return (x & (capacity_ - 1)); }
  int push(void* p) {
    int err = -EAGAIN;
    uint64_t push_limit = AL(&pop_) + capacity_;
    uint64_t old_push = 0;
    uint64_t push = AL(&push_);
    while((old_push = push) < push_limit
          && old_push != (push = VCAS(&push_, old_push, old_push + 1)))
    {
      PAUSE();
    }
    if (push < push_limit)
    {
      void** pdata = items_ + idx(push);
      while(!CAS(pdata, NULL, p))
      {
        PAUSE();
      }
      err = 0;
    }
    return err;
  }
  int pop(void*& p) {
    int err = -EAGAIN;
    uint64_t pop_limit = AL(&push_);
    uint64_t old_pop = 0;
    uint64_t pop = AL(&pop_);
    while((old_pop = pop) < pop_limit
           && old_pop != (pop = VCAS(&pop_, old_pop, old_pop + 1)))
    {
      PAUSE();
    }
    if (pop < pop_limit)
    {
      void** pdata = items_ + idx(pop);
      while(NULL == (p = FAS(pdata, NULL)))
      {
        PAUSE();
      }
      err = 0;
    }
    return err;
  }
  int64_t remain() const { return push_ - pop_; }
private:
  uint64_t count_ CACHE_ALIGNED;
  uint64_t push_ CACHE_ALIGNED;
  uint64_t pop_ CACHE_ALIGNED;
  uint64_t capacity_ CACHE_ALIGNED;
  void** items_;
};


class SlowSpinQueue
{
public:
  SlowSpinQueue(): push_(0), pop_(0), capacity_(0), items_(NULL) {}
  ~SlowSpinQueue(){}
  int init(uint64_t capacity, void** items) {
    int err = 0;
    capacity_ = capacity;
    memset(items, 0, sizeof(void*) * capacity);
    items_ = items;
    return err;
  }
  int destroy() {
    int err = 0;
    items_ = NULL;
    return err;
  }
  int64_t idx(int64_t x) { return x & (capacity_ - 1); }
  int push(void* p) {
    int err = -EAGAIN;
    uint64_t push = 0;
    while((push = AL(&push_)) < AL(&pop_) + capacity_)
    {
      if (!CAS(&push_, push, push + 1))
      {
        PAUSE();
      }
      else
      {
        err = 0;
        break;
        while(!CAS(items_ + idx(push), NULL, p))
        {
          PAUSE();
        }
        err = 0;
        break;
      }
    }
    return err;
  }
  int pop(void*& p) {
    int err = -EAGAIN;
    uint64_t pop = 0;
    while((pop = AL(&pop_)) < AL(&push_))
    {
      if (!CAS(&pop_, pop, pop + 1))
      {
        PAUSE();
      }
      else
      {
        err = 0;
        break;
        void** pdata = items_ + idx(pop);
        while(NULL == (p = AL(pdata)) || !CAS(pdata, p, NULL))
        {
          PAUSE();
        }
        err = 0;
        break;
      }
    }
    return err;
  }
  int64_t remain() const { return push_ - pop_; }
private:
  uint64_t count_ CACHE_ALIGNED;
  uint64_t push_ CACHE_ALIGNED;
  uint64_t pop_ CACHE_ALIGNED;
  uint64_t capacity_ CACHE_ALIGNED;
  void** items_;
};

typedef SlowSpinQueue SpinQueue;
