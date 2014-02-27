#ifndef __OB_AX_FIXED_HASH_SET_H__
#define __OB_AX_FIXED_HASH_SET_H__
#include "fixed_array.h"
#include "obj_pool.h"

template<typename T>
class FixedHashSet
{
public:
  struct Slot
  {
    Slot(int64_t seq): lock_(), head_(NULL) { UNUSED(seq); }
    ~Slot() {}
    SpinLock lock_;
    void* head_;
  };
  struct Item
  {
    Item(): next_(NULL), data_() {}
    ~Item() {}
    Item* next_;
    T data_;
  };
public:
  int init(int64_t capacity) {
    int err = AX_SUCCESS;
    if (AX_SUCCESS != (err = slot_array_.init(capacity)))
    {}
    else if (AX_SUCCESS != (err = item_pool_.init(capacity)))
    {}
    return err;
  }
  int find(T& item) {
    int err = AX_NOT_EXIST;
    Slot* slot = NULL;
    if (NULL == (slot = slot_array_.get(item.hash())))
    {
      err = AX_NOT_INIT;
    }
    else
    {
      SpinLock::Guard guard(slot->lock_);
      for (Item* cur_item = slot->head_; NULL != cur_item; cur_item = cur_item->next_)
      {
        if (cur_item->equal(item))
        {
          item.assign(*cur_item);
          err = AX_SUCCESS;
          break;
        }
      }
    }
    return err;
  }
  int add(T& item) {
    int err = AX_NOT_EXIST;
    Slot* slot = NULL;
    if (NULL == (slot = slot_array_.get(item.hash())))
    {
      err = AX_NOT_INIT;
    }
    else
    {
      SpinLock::Guard guard(slot->lock_);
      for (Item* cur_item = slot->head_; NULL != cur_item; cur_item = cur_item->next_)
      {
        if (cur_item->equal(item))
        {
          cut_item->assign(item);
          err = AX_SUCCESS;
          break;
        }
      }
      Item* new_item = NULL;
      if (AX_SUCCESS == err)
      {}
      else if (NULL == (new_item = item_pool_.alloc()))
      {
        err = AX_HASH_SIZE_OVERFLOW;
      }
      else
      {
        new_item->assign(item);
        new_item->next_ = slot->head_;
        slot->head_ = new_item;
      }
    }
    return err;
  }
  int del(T& item) {
    int err = AX_NOT_EXIST;
    Slot* slot = NULL;
    if (NULL == (slot = slot_array_.get(item.hash())))
    {
      err = AX_NOT_INIT;
    }
    else
    {
      SpinLock::Guard guard(slot->lock_);
      for (Item* cur_item = slot->head_, *prev_ptr = &slot->head_; NULL != cur_item; prev_ptr = &cur_item->next_, cur_item = cur_item->next_)
      {
        if (cur_item->equal(item))
        {
          item.assign(*cur_item);
          *prev_ptr = item->next_;
          err = AX_SUCCESS;
          break;
        }
      }
    }
    return err;
  }
private:
  int64_t len_;
  Slot* slots_;
  Item* items_;
  SpinQueue item_;
};

#endif /* __OB_AX_FIXED_HASH_SET_H__ */
