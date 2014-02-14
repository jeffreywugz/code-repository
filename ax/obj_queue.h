#ifndef __OB_AX_OBJ_QUEUE_H__
#define __OB_AX_OBJ_QUEUE_H__
#include "obj_pool.h"
#include "futex_queue.h"

template <typename T>
class ObjQueue
{
public:
  ObjQueue(){}
  ~ObjQueue(){}
public:
  int init(int64_t qsize) {
    int err = AX_SUCCESS;
    if (qsize <= 0)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (AX_SUCCESS != (err = pool_.init(qsize)))
    {}
    else if (AX_SUCCESS != (err = queue_.init(qsize)))
    {}
    return err;
  }
  int push(T* obj, const timespec* timeout) {
    int err = AX_SUCCESS;
    T* inner_obj = NULL;
    if (NULL == obj)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL == (inner_obj = pool_.alloc()))
    {
      err = OB_QUEUE_OVERFLOW;
    }
    else
    {
      *inner_obj = obj;
      err = queue_.push(inner_obj);
    }
    return err;
  }
  int pop(T* obj, const timespec* timeout) {
    int err = AX_SUCCESS;
    T* inner_obj = NULL;
    if (NULL == obj)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (AX_SUCCESS != (err = queue_.pop(inner_obj, timeout)))
    {}
    else
    {
      *obj = *inner_obj;
      pool_.free(inner_obj)
    }
    return err;
  }
private:
  ObjPool<T> pool_;
  FutexQueue queue_;
};

#endif /* __OB_AX_OBJ_QUEUE_H__ */
