#define wait_with_backoff(expr) while(expr){PAUSE(); }

class MCSLock
{
public:
  MCSLock(): tail_(NULL) {}
  ~MCSLock() {}
  struct Node
  {
    Node(): next_(NULL), locked_(1) {}
    ~Node(){}
    void reset() { next_ = NULL; locked_ = 1; }
    bool is_locked() { return locked_ != 0; }
    Node* volatile next_;
    volatile int64_t locked_;
  };
  void lock(Node* node)
  {
    Node* prev = __sync_lock_test_and_set(&tail_, node);
    if (NULL != prev)
    {
      prev->next_ = node;
      wait_with_backoff(node->is_locked());
    }
  }
  void unlock(Node* node)
  {
    if (NULL == node->next_)
    {
      if (__sync_bool_compare_and_swap(&tail_, node, NULL))
      {
        return;
      }
      wait_with_backoff(NULL == node->next_);
    }
    node->next_->locked_ = 0;
  }
private:
  Node* tail_ CACHE_ALIGNED;
};
