#ifndef __OB_AX_ID_MAP_H__
#define __OB_AX_ID_MAP_H__

#include "spin_queue.h"
#define IDMAP_INVALID_ID 0

class IDMap
{
  typedef int64_t id_t;
  struct Item
  {
    Item(int64_t seq): lock_(), seq_(seq), data_(NULL) {}
    ~Item(){}
    SpinLock lock_;
    int64_t seq_;
    void *data_;
  };
public:
  IDMap(){}
  ~IDMap(){}
public:
  int init(int64_t num) {
    int err = AX_SUCCESS;
    if (AX_SUCCESS != (err = free_list_.init(num)))
    {}
    else if (AX_SUCCESS != (err = map_.init(num)))
    {}
    for(int64_t i = 0; AX_SUCCESS != err && i < num; i++)
    {
      err = free_list_.push(map_.get(i));
    }
    return err;
  }
public:
  int assign(id_t& seq, void *value) {
    int err = AX_SUCCESS;
    Item* item = NULL;
    if (NULL == value)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (AX_SUCCESS != (err = free_list_.pop(item)))
    {}
    else
    {
      SpinLock::Guard guard(item->lock_);
      item->data_ = value;
      seq = item->seq_;
    }
    return err;
  }
  int erase(id_t id) {
    int err = AX_SUCCESS;
    Item* item = NULL;
    if (NULL == (item = map_.get(id)))
    {
      err = AX_NOT_INIT;
    }
    else
    {
      SpinLock::Guard guard(item->lock_);
      if (id != item->seq_)
      {
        err = AX_NOT_EXIST;
      }
      else
      {
        item->data_ = NULL;
        item->seq_ += map_.len();
      }
    }
    return err;
  }
  int get(id_t id, void*& data) const {
    int err = AX_SUCCESS;
    Item* item = NULL;
    if (NULL == (item = map_.get(id)))
    {
      err = AX_NOT_INIT;
    }
    else
    {
      item->lock_.lock();
      if (id != item->seq_)
      {
        err = AX_NOT_EXIST;
        item->lock_.unlock();
      }
      else
      {
        data = item->data_;
      }
    }
    return err;
  }
  int revert(id_t id) {
    int err = AX_SUCCESS;
    Item* item = NULL;
    if (NULL == (item = map_.get(id)))
    {
      err = AX_NOT_INIT;
    }
    else
    {
      if (id != item->seq_)
      {
        err = AX_NOT_EXIST;
      }
      else
      {
        item->lock_.unlock();
      }
    }
    return err;
  }
private:
  SpinQueue free_list_;
  FixedArray<Item> map_;
};

