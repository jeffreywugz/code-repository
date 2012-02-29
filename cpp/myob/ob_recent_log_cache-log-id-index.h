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

#ifndef __OCEANBASE_UPDATE_SERVER_OB_RECENT_LOG_CACHE_H__
#define __OCEANBASE_UPDATE_SERVER_OB_RECENT_LOG_CACHE_H__
#include "common/ob_define.h"
#include "common/ob_log_cursor.h"
#include "ob_fixed_ring_buffer.h"
#include "ob_versioned_recent_cache.h"

using oceanbase::common::ObLogCursor;
namespace oceanbase
{
  namespace updateserver
  {
    typedef ObVersionedRecentCache<int64_t, int64_t> ObLogPosIndex;
    class ObRecentLogCache
    {
      public:
        ObRecentLogCache();
        ~ObRecentLogCache();
        int init(int64_t log_buf_len, int64_t n_indexes);
        int push_log(const int64_t start_id, const int64_t end_id, const char* buf, const int64_t len);
        int get_log(const int64_t start_id, int64_t& end_id, char* buf, const int64_t len, int64_t& read_count);
        int reset();
        int get_remain_buf_size(const int64_t log_id, int64_t& size); // 覆盖log_id的日志之前，还剩余的buf。
      protected:
        bool is_inited() const;
        int check_state() const;
      private:
        DISALLOW_COPY_AND_ASSIGN(ObRecentLogCache);
        bool is_inited_;
        volatile int64_t version_;
        ObFixedRingBuffer log_buf_;
        ObLogPosIndex log_pos_index_;
        int64_t end_id_;
    };
  } // end namespace updateserver
} // end namespace oceanbase
#endif //__OCEANBASE_UPDATE_SERVER_OB_RECENT_LOG_CACHE_H__
