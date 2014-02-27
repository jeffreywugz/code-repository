#ifndef __OB_AX_ID_MAP_H__
#define __OB_AX_ID_MAP_H__

#include "spin_queue.h"
#define IDMAP_INVALID_ID 0

struct IdLock
{
  IdLock(id_t id): rwlock_(), id_(id) { rwlock_.wrlock(); }
  ~IdLock() {}
  id_t born() {
    id_t id = AL(id_);
    rwlock_.wrunlock();
    return id;
  }
  int inc_ref(id_t id) {
    int err = AX_SUCCESS;
    if (id != AL(&id_))
    {
      err = AX_ID_NOT_MATCH;
    }
    else if (!rwlock_.try_rdlock())
    {
      err = AX_STATE_NOT_MATCH;
    }
    else if (id != AL(&id_))
    {
      err = AX_ID_NOT_MATCH;
      rwlock_.rdunlock();
    }
    return err;
  }
  int dec_ref(id_t id) {
    int err = AX_SUCCESS;
    if (id != AL(&id_))
    {
      err = AX_ID_NOT_MATCH;
    }
    else
    {
      rwlock_.rdunlock();
    }
    return err;
  }
  int reclaim(id_t old_id, id_t new_id) {
    int err = AX_SUCCESS;
    if (old_id != AL(&id_))
    {
      err = AX_ID_NOT_MATCH;
    }
    else
    {
      rwlock_.wrlock();
      if (old_id != AL(&id_))
      {
        err = AX_ID_NOT_MATCH;
        rwlock_.wrunlock();
      }
      else
      {
        AS(&id_, new_id);
      }
    }
    return err;
  }
  RWLock rwlock_;
  id_t id_;
};

class IDMap
{
  typedef uint64_t id_t;
  typedef void* value_t;
  struct Item
  {
    Item(int64_t seq): idlock_(seq), value_() {}
    ~Item(){}
    IdLock idlock_;
    value_t value_;
  };
public:
  IDMap(): capacity_(0), items_(NULL) {}
  ~IDMap(){ destroy(); }
public:
  int init(int64_t capacity, void* buf) {
    int err = AX_SUCCESS;
    MemChunkCutter cutter(calc_mem_usage(capacity), buf);
    if (capacity <= 0 || !is2n(capacity) || NULL == buf)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (AX_SUCCESS != (err = free_list_.init(capacity, cutter.alloc(free_list_.calc_mem_usage(capacity)))))
    {}
    else
    {
      capacity_ = capacity;
      items_ = cutter.alloc(sizeof(Item) * capacity_);
      memset(items_, 0, capacity * sizeof(Item));
    }
    for(int64_t i = 0; AX_SUCCESS == err && i < capacity; i++)
    {
      new(items_ + i)Item(i);
      err = free_list_.push(items_ + i);
    }
    if (AX_SUCCESS != err)
    {
      destroy();
    }
    return err;
  }
  void destroy() {
    if (NULL != items_)
    {
      free_list_.destroy();
      capacity_ = 0;
      items_ = NULL;
    }
  }
  int64_t calc_mem_usage(int64_t capacity) { return free_list_.calc_mem_usage(capacity) + sizeof(Item) * capacity; }
  int64_t idx(int64_t x) { return x & (capacity_ - 1); }
public:
  int add(id_t& seq, value_t value) {
    int err = AX_SUCCESS;
    Item* item = NULL;
    if (AX_SUCCESS != (err = free_list_.pop(item)))
    {}
    else
    {
      item->value_ = value;
      err = item->idlock_.born();
    }
    return err;
  }
  int del(id_t id, value_t& value) {
    int err = AX_SUCCESS;
    Item* item = NULL;
    if (NULL == items_)
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS != (err = (item = items_ + idx(id))->idlock_.reclaim(id, id + capacity_)))
    {}
    else
    {
      value = item->value_;
      err = free_list_.push(item);
    }
    return err;
  }
  int fetch(id_t id, value_t& value) const {
    int err = AX_SUCCESS;
    Item* item = NULL;
    if (NULL == items_)
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS != (err = (item = items_ +idx(id))->idlock_.inc_ref(id)))
    {}
    else
    {
      value = item->value_;
    }
    return err;
  }
  int revert(id_t id) {
    int err = AX_SUCCESS;
    if (NULL == items_)
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS != (err = items_[idx(id)].idlock_.dec_ref(id)))
    {}
    return err;
  }
private:
  SpinQueue free_list_;
  int64_t capacity_;
  Item* items_;
};
