#ifndef __OB_AX_ID_MAP_H__
#define __OB_AX_ID_MAP_H__

#include "a0.h"
#include "spin_queue.h"
#include "lock.h"

struct IdLock
{
  IdLock(Id id): rwlock_(), id_(id) { rwlock_.wrlock(); }
  ~IdLock() {}
  Id born() {
    Id id = AL(&id_);
    rwlock_.wrunlock();
    return id;
  }
  int inc_ref_disregard_id(Id& id) {
    int err = AX_SUCCESS;
    if (!rwlock_.try_rdlock())
    {
      err = AX_STATE_NOT_MATCH;
    }
    else
    {
      id = AL(&id_);
    }
    return err;
  }
  int inc_ref(Id id) {
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
  int dec_ref(Id id) {
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
  int reclaim(Id old_id, Id new_id) {
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
  Id id_;
};

class IDMap
{
  typedef uint64_t Id;
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
  int init(int64_t capacity, char* buf) {
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
      items_ = (Item*)cutter.alloc(sizeof(Item) * capacity_);
      memset(items_, 0, capacity * sizeof(Item));
    }
    for(int64_t i = 0; AX_SUCCESS == err && i < capacity; i++)
    {
      new(items_ + i)Item(i + capacity);
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
  int64_t get_capacity() const { return capacity_; }
public:
  int add(Id& seq, value_t value) {
    int err = AX_SUCCESS;
    Item* item = NULL;
    if (AX_SUCCESS != (err = free_list_.pop((void*&)item)))
    {}
    else
    {
      item->value_ = value;
      seq = item->idlock_.born();
    }
    return err;
  }
  int del(Id id, value_t& value) {
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
  int fetch(Id id, value_t& value) {
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
  int revert(Id id) {
    int err = AX_SUCCESS;
    if (NULL == items_)
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS != (err = items_[idx(id)].idlock_.dec_ref(id)))
    {}
    return err;
  }
  int fetch_by_idx(Id _idx, Id& id, value_t& value) {
    int err = AX_SUCCESS;
    Item* item = NULL;
    if (NULL == items_)
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS != (err = (item = items_ +idx(_idx))->idlock_.inc_ref_disregard_id(id)))
    {}
    else
    {
      value = item->value_;
    }
    return err;
  }
private:
  SpinQueue free_list_;
  int64_t capacity_;
  Item* items_;
};
#endif /* __OB_AX_ID_MAP_H__ */
