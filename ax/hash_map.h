#ifndef __OB_AX_FIXED_HASH_SET_H__
#define __OB_AX_FIXED_HASH_SET_H__
#include "fixed_array.h"
#include "obj_pool.h"

class HashMap
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
    Item(): next_(NULL), key_(), value_() {}
    ~Item() {}
    Item* next_;
    key_t key_;
    value_t value_;
  };
public:
  HashMap(): capacity_(0), slots_(NULL) {}
  ~HashMap() { destroy(); }
  int64_t calc_mem_usage(int64_t capacity){ return free_items_.calc_mem_usage(capacity) + sizeof(Item) * capacity_ + sizeof(Slot) * capacity_; }
  int init(int64_t capacity, void* buf) {
    int err = AX_SUCCESS;
    MemChunkCutter cutter(calc_mem_usage(capacity), buf);
    if (capacity <= 0 || !is2n(capacity) || NULL == buf)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (AX_SUCCESS != (err = free_items_.init(capacity, cutter.alloc(free_items_.calc_mem_usage(capacity)))))
    {}
    else
    {
      Item* items = cutter.alloc(sizeof(Item) * capacity);
      for(int64_t i = 0; AX_SUCCESS == err && i < capacity; i++)
      {
        new(items + i)Item();
        err = free_items_.push(items + i);
      }
      if (AX_SUCCESS != err)
      {
        free_items_.destroy();
      }
      else if (AX_SUCCESS == err)
      {
        slots_ = cutter.alloc(sizeof(Slot) * capacity);
        for(int64_t i = 0; i < capacity; i++)
        {
          new(slots_ + i)Slot();
        }
        capacity_ = capacity;
      }
    }
    return err;
  }
  void destroy() {
    if (NULL != slots_)
    {
      capacity_ = 0;
      slots_ = NULL;
      free_items_.destroy();
    }
  }
  int find(key_t key,  value_t& value) {
    int err = AX_NOT_EXIST;
    if (NULL == slots_)
    {
      err = AX_NOT_INIT;
    }
    else
    {
      Slot* slot = slots + idx(hash(key));
      SpinLock::Guard guard(slot->lock_);
      for (Item* cur_item = slot->head_; NULL != cur_item; cur_item = cur_item->next_)
      {
        if (cur_item_->key_ == key)
        {
          value = cur_item->value_;
          err = AX_SUCCESS;
          break;
        }
      }
    }
    return err;
  }
  int add(key_t key, value_t value) {
    int err = AX_SUCCESS;
    if (NULL == slots_)
    {
      err = AX_NOT_INIT;
    }
    else
    {
      Slot* slot = slots + idx(hash(key));
      SpinLock::Guard guard(slot->lock_);
      for (Item* cur_item = slot->head_; NULL != cur_item; cur_item = cur_item->next_)
      {
        if (cur_item->key_ == key)
        {
          err = AX_ENTRY_EXIST;
          break;
        }
      }
      Item* new_item = NULL;
      if (AX_SUCCESS == err)
      {}
      else if (AX_SUCCESS != (err = free_items_.pop(new_item)))
      {}
      else
      {
        new_item->key_ = key;
        new_item->value_ = value;
        new_item->next_ = slot->head_;
        slot->head_ = new_item;
      }
    }
    return err;
  }
  int del(key_t key) {
    int err = AX_ENTRY_NOT_EXIST;
    if (NULL == slots_)
    {
      err = AX_NOT_INIT;
    }
    else
    {
      Slot* slot = slots + idx(hash(key));
      SpinLock::Guard guard(slot->lock_);
      Item* cur_item = slot->head_;
      Item *prev_ptr = &slot->head_;
      for (; NULL != cur_item; prev_ptr = &cur_item->next_, cur_item = cur_item->next_)
      {
        if (cur_item->key_ == key)
        {
          err = AX_SUCCESS;
          break;
        }
      }
      if (AX_SUCCESS == err)
      {
        *prev_ptr = cur_item->next_;
        err = free_items_.push(cur_item);
      }
    }
    return err;
  }
private:
  int64_t capacity_;
  Slot* slots_;
  SpinQueue free_items_;
};

#endif /* __OB_AX_FIXED_HASH_SET_H__ */
