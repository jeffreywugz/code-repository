#ifndef __OB_AX_LINK_QUEUE_H__
#define __OB_AX_LINK_QUEUE_H__
#include "a0.h"
#include "lock.h"

struct LinkNode
{
  LinkNode(): next_(NULL) {}
  ~LinkNode() {}
  LinkNode* next_;
};

class LinkQueue
{
public:
  typedef LinkNode Item;
  typedef SpinLock Lock;
  typedef SpinLock::Guard LockGuard;
public:
  LinkQueue(): head_(NULL), tail_(NULL) {}
  ~LinkQueue() {}
public:
  int push(Item* item) {
    int err = AX_SUCCESS;
    LockGuard guard(lock_);
    if (NULL == item)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL == tail_)
    {
      item->next_ = NULL;
      head_ = item;
      tail_ = item;
    }
    else
    {
      item->next_ = NULL;
      tail_->next_ = item;
      tail_ = item;
    }
    return err;
  }
  int pop(Item*& item) {
    int err = AX_SUCCESS;
    LockGuard guard(lock_);
    if (NULL == head_)
    {
      item = NULL;
      err = AX_EAGAIN;
    }
    else
    {
      item = head_;
      head_ = head_->next_;
      if (NULL == head_)
      {
        tail_ = NULL;
      }
    }
    return err;
  }
private:
  Lock lock_;
  Item* head_;
  Item* tail_;
};
#endif /* __OB_AX_LINK_QUEUE_H__ */
