
class SpinQueue
{
public:
  SpinQueue(): count_(0), push_(0), pop_(0), capacity_(0), items_(NULL) {}
  ~SpinQueue(){}
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
    uint64_t count = 0;
    while((count = AL(&count_)) < capacity_
          && !CAS(&count_, count, count + 1))
    {
      PAUSE();
    }
    if (count < capacity_)
    {
      uint64_t push = FAA(&push_, 1);
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
    uint64_t count = 0;
    while((count = AL(&count_)) > 0
          && !CAS(&count_, count, count - 1))
    {
      PAUSE();
    }
    if (count > 0)
    {
      uint64_t pop = FAA(&pop_, 1);
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

