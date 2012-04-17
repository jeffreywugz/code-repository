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
#include "ob_indexed_log_buffer.h"
#include "ob_log_buffer.h"

using namespace oceanbase::common;

namespace oceanbase
{
  namespace updateserver
  {
    int update_log_index_in_buf(const int64_t version, ObLogPosIndex& log_index, const int64_t start_pos,
                                     int64_t& start_id, int64_t& end_id, const char* log_data, int64_t data_len)
    {
      int err = OB_SUCCESS;
      ObLogEntry log_entry;
      int64_t pos = 0;
      int64_t old_pos = 0;
      start_id = 0;
      end_id = 0;
      while (OB_SUCCESS == err && pos < data_len)
      {
        old_pos = pos;
        if (OB_SUCCESS != (err = log_entry.deserialize(log_data, data_len, pos)))
        {
          TBSYS_LOG(ERROR, "log_entry.deserialize(log_data=%p, data_len=%ld, pos=%ld)=>%d", log_data, data_len, pos, err);
        }
        else if (OB_SUCCESS != (err = log_entry.check_data_integrity(log_data + pos)))
        {
          TBSYS_LOG(ERROR, "log_entry.check_data_integrity()=>%d", err);
        }
        else if (OB_SUCCESS != (err = log_index.add(version, log_entry.seq_, start_pos + old_pos)))
        {
          TBSYS_LOG(ERROR, "log_index.add(version=%ld, log_id=%ld, pos=%ld)=>%d",
                    version, log_entry.seq_, start_pos + old_pos, err);
        }
        else
        {
          pos += log_entry.get_log_data_len();
          end_id = log_entry.seq_ + 1;
          if (0 >= start_id)
          {
            start_id = log_entry.seq_;
          }
        }
      }
      return err;
    }

    ObIndexedLogBufferReader::ObIndexedLogBufferReader() : log_buf_(NULL), log_index_()
    {}

    ObIndexedLogBufferReader::~ObIndexedLogBufferReader()
    {}

    bool ObIndexedLogBufferReader::is_inited() const
    {
      return NULL == log_buf_;
    }

    int ObIndexedLogBufferReader::init(ObLogBuffer* log_buf)
    {
      int err = OB_SUCCESS;
      if (is_inited())
      {
        err = OB_INIT_TWICE;
      }
      else if (NULL == log_buf)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else
      {
        log_buf_ = log_buf;
      }
      return err;
    }
    
    int ObIndexedLogBufferReader::get_log(const int64_t start_id, int64_t& end_id, char* buf, const int64_t len, int64_t& read_count)
    {
      int err = OB_SUCCESS;
      int64_t real_start_id = 0;
      int64_t version = 0;
      int64_t pos = 0;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (0 > (version = log_buf_->get_start_pos()))
      {
        err = OB_ERR_UNEXPECTED;
      }
      else if (OB_SUCCESS == err
          && OB_SUCCESS != (err = log_index_.get(version, start_id, pos))
          && OB_EAGAIN != err)
      {
        TBSYS_LOG(ERROR, "log_index_.get()=>%d", err);
      }
      else if (OB_EAGAIN == err && OB_SUCCESS != (err = log_buf_->seek(start_id, pos)))
      {
        if (OB_DATA_NOT_SERVE != err)
        {
          TBSYS_LOG(ERROR, "log_buf.seek(start_id=%ld)=>%d", start_id, err);
        }
      }
      else if (OB_SUCCESS != (err = log_buf_->get_log(pos, real_start_id, end_id, buf, len, read_count)))
      {
        TBSYS_LOG(ERROR, "log_buf_.get_log(pos=%ld, buf=%p[%ld])=>%d", pos, buf, len, err);
      }
      else if (start_id != real_start_id)
      {
        err = OB_ERR_UNEXPECTED;
      }
      else if (OB_SUCCESS != (err = log_index_.add(version, end_id, pos + read_count)))
      {
        TBSYS_LOG(ERROR, "log_index.add(version=%ld, log_id=%ld, pos=%ld)=>%d", version, end_id, pos+read_count);
      }
      return err;
    }    
  }; // end namespace updateserver
}; // end namespace oceanbase

