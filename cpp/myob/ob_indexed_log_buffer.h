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
#ifndef __OB_UPDATESERVER_OB_INDEXED_LOG_BUFFER_H__
#define __OB_UPDATESERVER_OB_INDEXED_LOG_BUFFER_H__
#include "ob_log_src.h"
#include "ob_versioned_recent_cache.h"

namespace oceanbase
{
  namespace updateserver
  {
    class ObLogBuffer;
    typedef ObVersionedRecentCache<int64_t, int64_t> ObLogPosIndex;
    class ObIndexedLogBufferReader : public IObLogSrc
    {
      public:
        ObIndexedLogBufferReader();
        ~ObIndexedLogBufferReader();
        int init(ObLogBuffer* log_buf);
        virtual int get_log(const int64_t start_id, int64_t& end_id, char* buf, const int64_t len, int64_t& read_count);
      protected:
        bool is_inited() const;
      private:
        ObLogBuffer* log_buf_;
        ObLogPosIndex log_index_;
    };
  }; // end namespace updateserver
}; // end namespace oceanbase
#endif /* __OB_UPDATESERVER_OB_INDEXED_LOG_BUFFER_H__ */
