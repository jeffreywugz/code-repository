#ifndef __OB_AX_CACHE_INDEX_H__
#define __OB_AX_CACHE_INDEX_H__
#include "a0.h"
#include "lock.h"
#include "spin_queue.h"
#include "hash_func.h"

class CacheIndex
{
public:
  typedef Id key_t;
  typedef void* value_t;
  struct Item
  {
    Item(): lock_(), next_(NULL), key_(), value_() {}
    ~Item() {}
    SpinLock lock_;
    Item* next_;
    key_t key_;
    value_t value_;
  };
  struct Slot
  {
    Slot(): lock_(), head_(NULL) {}
    ~Slot() {}
    SpinLock lock_;
    Item* head_;
  };
  typedef SpinLock::Guard LockGuard;
  static uint32_t hash(key_t key) { return murmurhash2(&key, sizeof(key), 0); }
public:
  CacheIndex(): capacity_(0), slots_(NULL) {}
  ~CacheIndex() { destroy(); }
  int64_t calc_mem_usage(int64_t capacity){ return free_items_.calc_mem_usage(capacity) + sizeof(Item) * capacity_ + sizeof(Slot) * capacity_; }
  int init(int64_t capacity, char* buf) {
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
      Item* items = (Item*)cutter.alloc(sizeof(Item) * capacity);
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
        slots_ = (Slot*)cutter.alloc(sizeof(Slot) * capacity);
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
  int64_t idx(int64_t x) { return x & (capacity_ - 1); }
  int lock(key_t key, value_t& value) {
    int err = AX_SUCCESS;
    if (NULL == slots_)
    {
      err = AX_NOT_INIT;
    }
    else
    {
      Slot* slot = slots_ + idx(hash(key));
      LockGuard guard(slot->lock_);
      Item* found_item = NULL;
      for (Item* cur_item = slot->head_; NULL != cur_item; cur_item = cur_item->next_)
      {
        if (cur_item->key_ == key)
        {
          found_item = cur_item;
          break;
        }
      }
      if (NULL != found_item)
      {}
      else if (AX_SUCCESS != (err = free_items_.pop((void*&)found_item)))
      {}
      else
      {
        found_item->value_ = value;
        found_item->next_ = slot->head_;
        slot->head_ = found_item;
      }
      if (NULL != found_item)
      {
        found_item->lock_.lock();
        value = found_item->value_;
      }
    }
    return err;
  }
  int unlock(key_t key, value_t value) {
    int err = AX_NOT_EXIST;
    if (NULL == slots_)
    {
      err = AX_NOT_INIT;
    }
    else
    {
      Slot* slot = slots_ + idx(hash(key));
      LockGuard guard(slot->lock_);
      for (Item* cur_item = slot->head_; NULL != cur_item; cur_item = cur_item->next_)
      {
        if (cur_item->key_ == key)
        {
          cur_item->value_ = value;
          cur_item->lock_.unlock();
          err = AX_SUCCESS;
          break;
        }
      }
    }
    return err;
  }
private:
  int64_t capacity_;
  Slot* slots_;
  SpinQueue free_items_;
};

#endif /* __OB_AX_CACHE_INDEX_H__ */
