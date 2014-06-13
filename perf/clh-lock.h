
class CLHLock
{
private:
  enum {MAX_THREAD_NUM = 256};
  struct Item {
    Item(): prev_(NULL), need_wait_(true) {}
    ~Item() {}
    Item* prev_;
    volatile bool need_wait_;
  } CACHE_ALIGNED;
public:
  CLHLock(): next_tid_(0), tail_(item_) {
    tail_->need_wait_ = false;
    pthread_key_create(&key_, NULL);
  }
  ~CLHLock() { pthread_key_delete(key_); }
  int lock() {
    Item* cur_item = NULL;
    if (NULL == (cur_item = (Item*)pthread_getspecific(key_)))
    {
      cur_item = item_ + __sync_add_and_fetch(&next_tid_, 1);
      pthread_setspecific(key_, cur_item);
    }
    cur_item->prev_ = __sync_lock_test_and_set(&tail_, cur_item);
    Item* prev = cur_item->prev_;
    while(prev->need_wait_)
    {
      PAUSE();
    }
    return 0;
  }
  int unlock() {
    Item* cur_item = (Item*)pthread_getspecific(key_);
    pthread_setspecific(key_, cur_item->prev_);
    cur_item->need_wait_ = false;
    return 0;
  }
private:
  pthread_key_t key_;
  uint64_t next_tid_ CACHE_ALIGNED;
  Item* tail_;
  Item item_[MAX_THREAD_NUM + 1];
};
