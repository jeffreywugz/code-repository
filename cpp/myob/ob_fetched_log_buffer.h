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
#ifndef __OB_UPDATESERVER_OB_FETCHED_LOG_BUFFER_H__
#define __OB_UPDATESERVER_OB_FETCHED_LOG_BUFFER_H__
#include "ob_log_src.h"
#include "ob_fixed_ring_buffer.h"

namespace oceanbase
{
  namespace updateserver
  {
    class ObPrefetchLogBuffer : public IObLogSrc, public ObFixedRingBuffer
    {
      public:
        ObFetchedLogBuffer();
        virtual ~ObFetchedLogBuffer();
        // get_log()检测到要读的日志不连续时，返回OB_DISCONTINUOUS_LOG,
        int get_log(const int64_t start_id, int64_t& end_id,
                    char* buf, const int64_t len, int64_t& read_count);
        // 追加的日志必须保证与缓冲区中已有的日志连续, pos参数用作校验
        // 保证追加日志的操作是串行化的, push_log()可能返回OB_EAGAIN
        int push_log(const int64_t pos, char* buf, const int64_t len);
        int reset();
      private:
        DISALLOW_COPY_AND_ASSIGN(ObFetchedLogBuffer);
        int64_t read_pos_;
        int64_t next_end_pos_;
    };
  }; // end namespace updateserver
}; // end namespace oceanbase

#endif /* __OB_UPDATESERVER_OB_FETCHED_LOG_BUFFER_H__ */
