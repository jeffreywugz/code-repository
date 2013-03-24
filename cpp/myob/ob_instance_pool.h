/**
 * (C) 2007-2010 Taobao Inc.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * Version: $Id$
 *
 * Authors:
 *   yuanqi <yuanqi.xhf@taobao.com>
 *     - some work details if you want
 */
#ifndef __OB_COMMON_OB_INSTANCE_POOL_H__
#define __OB_COMMON_OB_INSTANCE_POOL_H__
#include "ob_fixed_queue.h"
#include "ob_malloc.h"

namespace oceanbase
{
  namespace common
  {
    template<typename T>
    class ObInstancePool
    {
      public:
        ObInstancePool(): free_list_(), buf_(NULL) {}
        ~ObInstancePool()
        {
          destroy();
        }
        void destroy()
        {
          T* p = NULL;
          while(OB_SUCCESS == free_list_.pop(p))
          {
            p->~T();
          }
          if (NULL != buf_)
          {
            ob_free(buf_);
            buf_ = NULL;
          }
          free_list_.destroy();
        }
        int init(int64_t capacity)
        {
          int err = OB_SUCCESS;
          int64_t size = sizeof(T) * capacity;
          if (capacity <= 0)
          {
            err = OB_INVALID_ARGUMENT;
          }
          else if (NULL != buf_)
          {
            err = OB_INIT_TWICE;
          }
          else if (NULL == (buf_ = (T*)ob_malloc(size, ObModIds::OB_INSTANCE_POOL)))
          {
            err = OB_ALLOCATE_MEMORY_FAILED;
            TBSYS_LOG(ERROR, "ob_malloc(size=%ld)=>NULL", size);
          }
          else if (OB_SUCCESS != (err = free_list_.init(capacity)))
          {
            TBSYS_LOG(ERROR, "free_list_.init(%ld)=>%d", capacity, err);
          }
          for (int64_t i = 0; OB_SUCCESS == err && i < capacity; i++)
          {
            if (NULL == new(buf_ + i)T())
            {
              err = OB_ERR_UNEXPECTED;
              TBSYS_LOG(ERROR, "new(%p)T()=>NULL", buf_ + i);
            }
            else if (OB_SUCCESS != (err = free_list_.push(buf_ + i)))
            {
              TBSYS_LOG(ERROR, "free_list_.push(i=%ld)=>%d", i, err);
            }
          }
          if (OB_SUCCESS != err)
          {
            destroy();
          }
          return err;
        }
        T* alloc()
        {
          int err = OB_SUCCESS;
          T* p = NULL;
          if (OB_SUCCESS != (err = free_list_.pop(p)))
          {
            TBSYS_LOG(WARN, "free_list.pop()=>%d", err);
          }
          else
          {
            p->reset();
          }
          return p;
        }
        int free(T* p)
        {
          int err = OB_SUCCESS;
          if (OB_SUCCESS != (err = free_list_.push(p)))
          {
            TBSYS_LOG(ERROR, "free_list.push(%p)=>%d", p, err);
          }
          return err;
        }
        int64_t get_free_num() { return free_list_.get_total(); }
      private:
        ObFixedQueue<T> free_list_;
        T* buf_;
    };
  }; // end namespace common
}; // end namespace oceanbase

#endif /* __OB_COMMON_OB_INSTANCE_POOL_H__ */
