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

#ifndef OCEANBASE_UPDATESERVER_OB_POS_LOG_READER_H_
#define OCEANBASE_UPDATESERVER_OB_POS_LOG_READER_H_
#include "common/ob_define.h"

namespace oceanbase
{
  namespace common
  {
    class ObLogCursor;
  };
  namespace updateserver
  {
    class ObRepeatedLogReader;
    class ObSessionIdMgr;
    class ObPosLogReader
    {
      public:
        ObPosLogReader();
        virtual ~ObPosLogReader();
        int init(const char* log_dir, ObSessionIdMgr* session_mgr);
        virtual int get_log(const int64_t advised_session_id, int64_t& returned_session_id,
                            const common::ObLogCursor& start_cursor, common::ObLogCursor& end_cursor,
                            char* buf, const int64_t len, int64_t& read_count);
      protected:        
        bool is_inited() const;
        int check_state() const;
        int get_reader(ObSingleLogReader*& log_reader, const int64_t session_id) const;
      private:
        const char* log_dir_;
        int64_t n_log_readers_;
        ObRepeatedLogReader* log_readers_;
        ObSessionIdMgr* session_mgr_;
    };
  }; // end namespace updateserver
}; // end namespace oceanbase
#endif // OCEANBASE_UPDATESERVER_OB_POS_LOG_READER_H_
