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
#ifndef __OB_UPDATESERVER_OB_LOG_BUFFER_H__
#define __OB_UPDATESERVER_OB_LOG_BUFFER_H__

#include "ob_ring_data_buffer.h"

namespace oceanbase
{
  namespace updateserver
  {
    // 串行地append_log(), 并发地get_log()
    class ObLogBuffer : public ObRingDataBuffer
    {
      public:
        ObLogBuffer();
        virtual ~ObLogBuffer();
        // 返回OB_DISCONTINUOUS_LOG, 需要重置。
        // 返回OB_NEED_RETRY, 当前block剩余空间不够，这时需要重新获取end_pos, 重试。
        int append_log(const int64_t pos, const int64_t start_id, const int64_t end_id, const char* buf, const int64_t len);
        // 返回OB_DATA_NOT_SERVE, pos无效，需要重新定位,
        // 返回OB_FOR_PADDING，pos指向当前block的最后位置，需要跳过当前block，需要将pos加上read_count, 重试。
        int get_log(const int64_t pos, int64_t& start_id, int64_t& end_id, char* buf, const int64_t len, int64_t& read_count) const;
        // 返回OB_DATA_NOT_SERVE，缓冲区中暂时没有编号为log_id的日志
        int seek(const int64_t log_id, const int64_t advised_pos, int64_t& real_pos) const;
        int reset();
        int dump_for_debug() const;
        int64_t get_start_id() const;
        int64_t get_end_id() const;
      protected:
        int get_next_entry(const int64_t pos, int64_t& next_pos, int64_t& log_id) const;
      private:
        int64_t end_id_;
    };
    // OB_FOR_PADDING被处理，可能返回OB_DATA_NOT_SERVE
    int get_from_log_buffer(ObLogBuffer* log_buf, const int64_t advised_pos, int64_t& real_pos,
                                const int64_t start_id, int64_t& end_id, char* buf, const int64_t len, int64_t& read_count);
    // OB_DISCONTINUOUS_LOG/OB_NEED_RETRY被处理，不应该返回任何错误
    int append_to_log_buffer(ObLogBuffer* log_buf, const int64_t start_id, const int64_t& end_id,
                             const char* buf, const int64_t len);
  }; // end namespace updateserver
}; // end namespace oceanbase

#endif /* __OB_UPDATESERVER_OB_LOG_BUFFER_H__ */
