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
#ifndef __OB_UPDATESERVER_OB_FIXED_RING_BUFFER_H__
#define __OB_UPDATESERVER_OB_FIXED_RING_BUFFER_H__
#include "common/ob_define.h"

namespace oceanbase
{
  namespace updateserver
  {
    class ObFixedRingBuffer
    {
      public:
        ObFixedRingBuffer();
        ~ObFixedRingBuffer();
        int init(const int64_t log_buf_len);
        int read(const int64_t pos, char* buf, const int64_t len, int64_t& read_count);
        // 并发append()操作执行前需要用CAS指令争用next_end_pos_的修改权，这样保证互斥
        int append(const int64_t pos, const char* buf, const int64_t len);
        int64_t get_end_pos() const;
        int64_t get_next_end_pos() const;
    protected:
        bool is_inited() const;
      private:
        DISALLOW_COPY_AND_ASSIGN(ObFixedRingBuffer);
        volatile int64_t end_pos_;
        volatile int64_t next_end_pos_;
        int64_t buf_len_;
        char* buf_;
    };
  }; // end namespace updateserver
}; // end namespace oceanbase

#endif /* __OB_UPDATESERVER_OB_FIXED_RING_BUFFER_H__ */
