/*
 * (C) 2007-2010 Taobao Inc.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 *
 * Authors:
 *   yuanqi.xhf <yuanqi.xhf@taobao.com>
 *     - some work details if you want
 *
 */

#ifndef __OCEANBASE_COMMON_OB_ATOMIC_TYPE_H__
#define __OCEANBASE_COMMON_OB_ATOMIC_TYPE_H__
#include "common/ob_define.h"

namespace oceanbase
{
  namespace common
  {
    template<typename T>
    // 单个写者，多个读者
    struct AtomicType
    {
      volatile int64_t version_; // version_小于等于0时无效
      AtomicType(): version_(0)
      {}
      ~AtomicType()
      {}

      int copy(T& that)
      {
        int err = OB_SUCCESS;
        int64_t version = version_;
        if (0 >= version)
        {
          err = OB_EAGAIN;
        }
        else
        {
          that = *((T*)this);
        }
        if (OB_SUCCESS == err && version != version_)
        {
          err = OB_EAGAIN;
        }
        return err;
      }

      int set(T& that)
      {
        int err = OB_SUCCESS;
        int64_t version = version_;
        version_ = -1;
        *((T*)this) = that;
        version_ = version + 1;
        return err;
      }
    };  
  } // end namespace common
} // end namespace oceanbase
#endif __OCEANBASE_COMMON_OB_ATOMIC_TYPE_H__
