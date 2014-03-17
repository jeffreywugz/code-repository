#ifndef __OB_AX_OBJ_POOL_H__
#define __OB_AX_OBJ_POOL_H__
#include "common.h"
#include "spin_queue.h"

template<typename T>
class ObjPool
{
public:
  typedef SpinQueue FreeList;
public:
  ObjPool() {}
  ~ObjPool() { destroy(); }
public:
  int init(int64_t capacity, char* buf) {
    int err = AX_SUCCESS;
    if (capacity <= 0 || NULL == buf)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (AX_SUCCESS != (err = free_list_.init(capacity, buf)))
    {}
    else
    {
      T* obj = (T*)(buf + free_list_.calc_mem_usage(capacity));
      for(int64_t i = 0; AX_SUCCESS == err && i < capacity; i++)
      {
        if (AX_SUCCESS != (err = free_list_.push(obj + i)))
        {}
        else
        {
          new(obj)T();
        }
      }
    }
    if (AX_SUCCESS != err)
    {
      destroy();
    }
    return err;
  }
  int64_t calc_mem_usage(int64_t capacity) {
    return sizeof(T) * capacity + free_list_.calc_mem_usage(capacity);
  }
  void destroy() {
    int err = AX_SUCCESS;
    T* obj = NULL;
    while(AX_SUCCESS == (err = free_list_.pop((void*&)obj)))
    {
      obj->~T();
    }
  }
  T* alloc() {
    int err = AX_SUCCESS;
    T* obj = NULL;
    if (AX_SUCCESS != (err = free_list_.pop((void*&)obj)))
    {}
    else
    {
      obj->reset();
    }
    return obj;
  }
  void free(T* obj) {
    int err = AX_SUCCESS;
    if (AX_SUCCESS != (err = free_list_.push(obj)))
    {
      MLOG(ERROR, "free_list.push(%p)=>%d", obj, err);
    }
  }
private:
  FreeList free_list_;
};

#endif /* __OB_AX_OBJ_POOL_H__ */
