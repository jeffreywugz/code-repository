#ifndef __OB_AX_OBJ_POOL_H__
#define __OB_AX_OBJ_POOL_H__
#include "malloc.h"
#include "spin_queue.h"

template<typename T>
class ObjPool
{
public:
  typedef SpinQueue ObjQueue;
public:
  ObjPool() {}
  ~ObjPool() { destroy(); }
public:
  int init(int64_t count, int mod_id) {
    int err = AX_SUCCESS;
    if (count <= 0)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (AX_SUCCESS != (err = free_list_.init(count)))
    {}
    else if (NULL == (obj_buf_ = (T*)ax_malloc(sizeof(T) * count, mod_id)))
    {
      err = AX_ALLOCATE_MEMORY_FAILED;
    }
    for(int64_t i = 0; AX_SUCCESS == err && i < count; i++)
    {
      T* obj = obj_buf_ + i;
      if (AX_SUCCESS != (err = free_list_.push(obj)))
      {}
      else
      {
        new(obj)T();
      }
    }
    if (AX_SUCCESS != err)
    {
      destroy();
    }
    return err;
  }
  void destroy() {
    int err = AX_SUCCESS;
    T* obj = NULL;
    while(AX_SUCCESS == (err = free_list_.pop(obj)))
    {
      obj->~T();
    }
    if (NULL != obj_buf_)
    {
      ax_free(obj_buf_);
      obj_buf_ = NULL;
    }
  }
  T* alloc() {
    int err = AX_SUCCESS;
    T* obj = NULL;
    if (AX_SUCCESS != (err = free_list_.pop(obj)))
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
      AX_LOG(ERROR, "free_list.push(%p)=>%d", obj, err);
    }
  }
private:
  T* obj_buf_;
  ObjQueue free_list_;
};

#endif /* __OB_AX_OBJ_POOL_H__ */
