class Stack
{
  public:
    Stack(): capacity_(0), items_(NULL), top_(0) {}
    ~Stack() {}

    int init(void** items, int64_t capacity) {
      items_ = items;
      capacity_ = capacity;
    }

    int push(void* p) {
      int64_t top = top_;
      if (NULL == items_ || top&1 || top >= (capacity_<<1) || !__sync_bool_compare_and_swap(&top_, top, top+1))
        return -EAGAIN;
      items_[(top>>1)%capacity_] = p;
      top_++;
      return 0;
    }

    int pop(void*& p) {
      int64_t top = top_;
      if (NULL == items_ || top&1 || top < 2  _ || !__sync_bool_compare_and_swap(&top_, top, top-1))
        return -EAGAIN;
      p = items_[(top>>1)%capacity_];
      top_--;
      return 0;
    }
  private:
    int64_t capacity_;
    void** items_;
    volatile int64_t top_;
};
