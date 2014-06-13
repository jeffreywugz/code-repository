
class FastSpinLock
{
  enum {MAX_THREAD_NUM = 256};
  struct Item {
    Item(): tid_(0), seq_(0) {}
    ~Item() {}
    union {
      struct {
        uint32_t tid_;
        uint32_t seq_;
      };
      uint64_t value_;
    };
    char align_[] CACHE_ALIGNED;
  } CACHE_ALIGNED;
  public:
    FastSpinLock() { next_tid_ = 0; tail_.seq_ = ~0; pthread_key_create(&key_, NULL); }
    ~FastSpinLock() { pthread_key_delete(key_); }
    int lock() {
      Item* cur_item = NULL;
      Item prev;
      if (NULL == (cur_item = (Item*)pthread_getspecific(key_)))
      {
        uint64_t tid = __sync_fetch_and_add(&next_tid_, 1);
        cur_item = item_ + tid;
        cur_item->tid_ = tid;
        pthread_setspecific(key_, cur_item);
      }
      prev.value_ = __sync_lock_test_and_set(&tail_.value_, cur_item->value_);
      volatile uint64_t& value = item_[prev.tid_].value_;
      while(value == prev.value_)
      {
        PAUSE();
      }
      return 0;
    }
  int unlock() {
    Item* cur_item = NULL;
    cur_item = (Item*)pthread_getspecific(key_);
    __sync_fetch_and_add(&cur_item->seq_, 1);
    return 0;
  }
  private:
    pthread_key_t key_;
    uint64_t next_tid_ CACHE_ALIGNED;
    Item tail_;
    Item item_[MAX_THREAD_NUM];
};
