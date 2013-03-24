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

#ifndef __OB_COMMON_OB_PIPE_BUFFER_H__
#define __OB_COMMON_OB_PIPE_BUFFER_H__
#include "ob_define.h"
#include "tbsys.h"

namespace oceanbase
{
  namespace common
  {
    class ObPipeBuffer
    {
      // 单线程push，单线程pop，pop线程处理完数据之后需要调用consumed()下次才能拿到新数据
      public:
        ObPipeBuffer(): buf_(NULL), capacity_(0), push_(0), pop_(0) {}
        ~ObPipeBuffer(){}
      public:
        int init(int64_t capacity, char* buf);
        int push(char* buf, const int64_t len, const int64_t timeout_us);
        //  处理完之后需要调用consumed()通知pipe_buffer，这样下次pop的时候才能拿到新数据
        int push_done(char* buf, const int64_t len);
        int pop(char* buf, const int64_t limit, int64_t& read_count, const int64_t timeout_us);
        int consumed(char* buf, const int64_t len);
      private:
        int push(char* buf, const int64_t len);
        int pop(char* buf, const int64_t limit, int64_t& read_count);
      private:
        DISALLOW_COPY_AND_ASSIGN(ObPipeBuffer);
      protected:
        char* buf_;
        int64_t capacity_;
        tbsys::CThreadCond cond_;
        volatile int64_t push_ CACHE_ALIGNED;
        volatile int64_t pop_ CACHE_ALIGNED;
    };
  }; // end namespace common
}; // end namespace oceanbase
#endif /* __OB_COMMON_OB_PIPE_BUFFER_H__ */
