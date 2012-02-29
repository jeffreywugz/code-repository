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

#include "common/ob_log_entry.h"
#include "common/data_buffer.h"
#include "common/utility.h"
#include "common/ob_repeated_log_reader.h"
#include "ob_pos_log_reader.h"
#include "ob_ups_log_utils.h"

using namespace oceanbase::common;
namespace oceanbase
{
  namespace updateserver
  {
    ObPosLogReader::ObPosLogReader(): log_dir_(NULL), n_log_readers_(0), log_readers_(NULL), session_mgr_(NULL)
    {
    }

    ObPosLogReader::~ObPosLogReader()
    {
      if (NULL != log_readers__)
      {
        delete []log_readers_;
      }
    }

    bool ObPosLogReader::is_inited() const
    {
      return NULL != log_dir_ && NULL != log_readers_ && n_log_readers_ > 0 && NULL != session_mgr_;
    }
    
    int ObPosLogReader::check_state() const
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "not init.");
      }
      return err;
    }

    int ObPosLogReader::init(const char* log_dir, ObSessionIdMgr* session_mgr)
    {
      int err = OB_SUCCESS;
      if (is_inited())
      {
        err = OB_INIT_TWICE;
        TBSYS_LOG(ERROR, "init twice.");
      }
      else if (NULL == log_dir || NULL == session_mgr)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "init(log_dir=%s, session_mgr=%p): invalid argument",
                  log_dir, session_mgr);
      }
      else if (OB_SUCCESS != (err = session_mgr->get_capacity(n_log_readers_)))
      {
        TBSYS_LOG(ERROR, "session_mgr->get_capacity()=>%d", err);
      }
      else if (NULL == (log_readers_ = new(std::nothrow) ObRepeatedLogReader[n_log_readers_]))
      {
        err = OB_ALLOCATE_MEMORY_FAILED ;
        TBSYS_LOG(ERROR, "new ObRepeatedLogReader[n_reader=%ld] failed.", n_log_readers_);
      }

      for(int64_t i = 0; OB_SUCCESS == err && i < n_log_readers_; i++)
      {
        if (OB_SUCCESS != (err = log_readers_[i].init(log_dir)))
        {
          TBSYS_LOG(ERROR, "log_reader[%ld].init(log_dir=%s)=>%d", i, log_dir, err);
        }
      }
      
      if (OB_SUCCESS == err)
      {
        log_dir_ = log_dir;
      }
      return err;
    }

    int ObPosLogReader::get_reader(ObSingleLogReader*& log_reader, const int64_t session_id) const
    {
      int err = OB_SUCCESS;
      ObRepeatedLogReader* log_reader = NULL;
      returned_session_id = 0;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (0 >= session_id)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "session_id[%ld] <= 0", session_id);
      }
      else
      {
        log_reader = log_readers_ + session_id % n_log_readers_;
      }
      return err;
    }

    int ObPosLogReader::get_log(const int64_t advised_session_id, int64_t& returned_session_id,
                                const common::ObLogCursor& start_cursor, common::ObLogCursor& end_cursor,
                                char* buf, const int64_t len, int64_t& read_count)
    {
      int err = OB_SUCCESS;
      int64_t pos = 0;
      ObRepeatedLogReader* log_reader = NULL;
      returned_session_id = 0;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (NULL == buf || 0 >= len || !start_cursor.is_valid())
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "get_log(start_cursor=%s, buf=%p, len=%ld)=>%d", start_cursor.to_str(), buf, len, err);
      }
      else if (OB_SUCCESS != (err = session_mgr->acquire_session(advised_session_id, return_session_id)))
      {
        TBSYS_LOG(ERROR, "acquire_session(advised_session_id=%ld)=>%d", advised_session_id, err);
      }
      else if (OB_SUCCESS != (err = get_reader(log_reader, returned_session_id)))
      {
        log_reader = log_readers_ + returned_session_id% n_log_readers_;
      }
      else if (OB_SUCCESS != (err = require_log_reader_seek_to(log_reader, start_cursor)))
      {
        if (OB_READ_NOTHING != err)
        {
          TBSYS_LOG(ERROR, "require_log_reader_seek_to(log_reader[%ld], start_cursor=%s)=>%d",
                    returned_sesseion_id, start_cursor.to_str(), err);
        }
      }
      else if (OB_SUCCESS != (err = read_multiple_logs(buf, len, pos, log_reader->log_reader_, start_cursor, end_cursor)))
      {
        if (OB_READ_NOTHING != err)
        {
          TBSYS_LOG(ERROR, "read_multiple_logs()=>%d", err);
        }
      }
      else
      {
        read_count = pos;
      }

      if (0 < returned_session_id && OB_SUCCESS != (err = session_mgr_->release_session(returned_session_id)))
      {
        TBSYS_LOG(ERROR, "release_session(session_id=%ld)=>%d", returned_session_id, err);
      }
      return err;
    }
  } // end namespace updateserver
} // end namespace oceanbase
