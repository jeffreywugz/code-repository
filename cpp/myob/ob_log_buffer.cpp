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
#include "common/utility.h"
#include "tbsys.h"
#include "ob_log_buffer.h"
#include "ob_ups_log_utils.h"

using namespace oceanbase::common;

namespace oceanbase
{
  namespace updateserver
  {
    ObLogBuffer::ObLogBuffer(): end_id_(0)
    {}

    ObLogBuffer::~ObLogBuffer()
    {}

    int ObLogBuffer::reset()
    {
      int err = OB_SUCCESS;
      TBSYS_LOG(INFO, "reset()");
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else
      {
        end_id_ = 0;
        start_pos_ = end_pos_;
      }
      return err;
    }

    int64_t ObLogBuffer::get_start_id() const
    {
      int err = OB_SUCCESS;
      int64_t next_pos = 0;
      int64_t log_id = 0;
      int64_t start_id = 0;
      if (OB_SUCCESS != (err = get_next_entry(start_pos_, next_pos, log_id))
        && OB_DATA_NOT_SERVE != err)
      {
        TBSYS_LOG(ERROR, "get_next_entry(pos=%ld)=>%d", start_pos_, err);
      }
      else if (OB_SUCCESS == err)
      {
        start_id = log_id;
      }
      return start_id;
    }

    int64_t ObLogBuffer::get_end_id() const
    {
      return end_id_;
    }

    int ObLogBuffer::dump_for_debug() const
    {
      int err = OB_SUCCESS;
      TBSYS_LOG(INFO, "pos=[%ld, %ld), id=[%ld, %ld)", start_pos_, end_pos_, get_start_id(), end_id_);
      ObRingDataBuffer::dump_for_debug();
      return err;
    }

    int ObLogBuffer::append_log(const int64_t pos, const int64_t start_id, const int64_t end_id, const char* buf, const int64_t len)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (NULL == buf || 0 > len || 0 >= start_id || start_id > end_id)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "append_log(range=[%ld,%ld], buf=%p[%ld]): invalid argument", start_id, end_id, buf, len);
      }
      else if (end_id_ > 0 && end_id_ != start_id)
      {
        err = OB_DISCONTINUOUS_LOG;
      }
      else if (OB_SUCCESS != (err = append(pos, buf, len)))
      {
        if (OB_BUF_NOT_ENOUGH != err)
        {
          TBSYS_LOG(ERROR, "append(pos=%ld, buf=%p[%ld])=>%d", get_end_pos(), buf, len, err);
        }
        else
        {
          err = OB_NEED_RETRY;
        }
      }
      else
      {
        end_id_ = end_id;
      }
      return err;
    }

    int ObLogBuffer:: get_next_entry(const int64_t pos, int64_t& next_pos, int64_t& log_id) const
    {
      int err = OB_SUCCESS;
      int64_t copy_count = 0;
      ObLogEntry log_entry;
      char log_entry_buf[sizeof(ObLogEntry)];
      int64_t log_entry_buf_pos = 0;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (OB_SUCCESS != (err = read(pos, log_entry_buf, sizeof(log_entry_buf), copy_count)))
      {
        if (OB_FOR_PADDING == err)
        {
          next_pos += copy_count;
        }
        else if (OB_DATA_NOT_SERVE != err)
        {
          TBSYS_LOG(ERROR, "read(pos=%ld, buf=%p[%ld])=>%d", pos, log_entry_buf, sizeof(log_entry_buf), err);
        }
      }
      else if (copy_count < (int64_t)sizeof(log_entry_buf))
      {
        err = OB_DATA_NOT_SERVE;
      }
      else if (OB_SUCCESS != (err = log_entry.deserialize(log_entry_buf, sizeof(log_entry_buf), log_entry_buf_pos)))
      {
        TBSYS_LOG(ERROR, "log_entry.deserialize(%p[%ld])=>%d", log_entry_buf, sizeof(log_entry_buf), err);
      }
      else
      {
        log_id = log_entry.seq_;
        next_pos = pos + log_entry.get_serialize_size() + log_entry.get_log_data_len();
      }
      return err;
    }

    int ObLogBuffer::seek(const int64_t log_id, const int64_t advised_pos, int64_t& real_pos) const
    {
      int err = OB_SUCCESS;
      int64_t cur_id = 0;
      int64_t next_pos = 0;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (log_id >= end_id_)
      {
        err = OB_DATA_NOT_SERVE;
      }
      else
      {
        real_pos = max(advised_pos, start_pos_);
      }
      while(OB_SUCCESS == err)
      {
        if (OB_SUCCESS != (err = get_next_entry(real_pos, next_pos, cur_id)))
        {
          if (OB_FOR_PADDING == err)
          {
            err = OB_SUCCESS;
            real_pos = next_pos;
          }
          else if (OB_DATA_NOT_SERVE != err)
          {
            TBSYS_LOG(ERROR, "get_next_entry(pos=%ld)=>%d", real_pos, err);
          }
        }
        else if (log_id < cur_id)
        {
          err = OB_DATA_NOT_SERVE;
        }
        else if (log_id == cur_id)
        {
          break;
        }
        else
        {
          real_pos = next_pos;
        }
      }
      return err;
    }

    int ObLogBuffer::get_log(const int64_t pos, int64_t& start_id, int64_t& end_id, char* buf, const int64_t len, int64_t& read_count) const
    {
      int err = OB_SUCCESS;
      int64_t copy_count = 0;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (NULL == buf || len <= 0 || 0 > pos)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "get_log(start_id=%ld, buf=%p[%ld]): invalid argument", start_id, buf, len);
      }
      else if (OB_SUCCESS != (err = read(pos, buf, len, copy_count)))
      {
        if (OB_DATA_NOT_SERVE != err || OB_FOR_PADDING != err)
        {
          TBSYS_LOG(ERROR, "copy_to_buf(%p[%ld], read_count=%ld)=>%d", buf, len, read_count, err);
        }
      }
      else if (OB_SUCCESS != (err = trim_log_buffer(buf, copy_count, read_count, start_id, end_id)))
      {
        TBSYS_LOG(ERROR, "parse_log_buffer(buf=%p[%ld], start_id=%ld)=>%d",
                  buf, copy_count, start_id, err);
      }
      return err;
    }

    int get_from_log_buffer(ObLogBuffer* log_buf, const int64_t advised_pos, int64_t& real_pos,
                                const int64_t start_id, int64_t& end_id, char* buf, const int64_t len, int64_t& read_count)
    {
      int err = OB_SUCCESS;
      int64_t real_start_id = start_id;
      if (NULL == log_buf || NULL == buf || start_id <= 0 || len <= 0)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else if (OB_SUCCESS != (err = log_buf->seek(start_id, advised_pos, real_pos)))
      {
        if (OB_DATA_NOT_SERVE != err)
        {
          TBSYS_LOG(ERROR, "log_buf->seek(start_id=%ld, advised_pos=%ld)=>%d", start_id, advised_pos, err);
        }
        else
        {
          TBSYS_LOG(DEBUG, "log_buf->seek(start_id=%ld, advised_pos=%ld): OB_DATA_NOT_SERVE", start_id, advised_pos);
        }
      }
      else if (OB_SUCCESS != (err = log_buf->get_log(real_pos, real_start_id, end_id, buf, len, read_count))
               && OB_FOR_PADDING != err)
      {
        if (OB_DATA_NOT_SERVE != err)
        {
          TBSYS_LOG(ERROR, "log_buf->get_log(pos=%ld, buf=%p[%ld])=>%d", real_pos, buf, len, err);
        }
        else
        {
          TBSYS_LOG(DEBUG, "log_buf->get_log(pos=%ld): OB_DATA_NOT_SERVE", real_pos);
        }
      }
      else if (OB_FOR_PADDING == err
               && OB_SUCCESS != (err = log_buf->get_log(real_pos += read_count, real_start_id, end_id, buf, len, read_count)))
      {
        TBSYS_LOG(ERROR, "log_buf->get_log(pos=%ld, buf=%p[%ld])=>%d", real_pos, buf, len, err);
      }
      else if (start_id != real_start_id)
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "start_id[%ld] != real_start_id[%ld]", start_id, real_start_id);
      }
      return err;
    }

    int append_to_log_buffer(ObLogBuffer* log_buf,
                             const int64_t start_id, const int64_t& end_id, const char* buf, const int64_t len)
    {
      int err = OB_SUCCESS;
      int reset_err = OB_SUCCESS;
      if (NULL == log_buf || NULL == buf || start_id <= 0 || end_id < start_id || len < 0)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "append_to_log_buffer(log_buf=%p, buf=%p[%ld], id=[%ld,%ld]): invalid argument",
                  log_buf, buf, len, start_id, end_id);
      }
      else if (OB_SUCCESS != (err = log_buf->append_log(log_buf->get_end_pos(), start_id, end_id, buf, len))
               && OB_NEED_RETRY != err && OB_DISCONTINUOUS_LOG != err)
      {
        TBSYS_LOG(ERROR, "log_buf->append_log(pos=%ld, range=[%ld, %ld], buf=%p[%ld])=>%d",
                  log_buf->get_end_pos(), start_id, end_id, buf, len, err);
      }
      else if (OB_SUCCESS == err)
      {}
      else if (OB_DISCONTINUOUS_LOG == err && OB_SUCCESS != (reset_err = log_buf->reset()))
      {
        err = reset_err;
        TBSYS_LOG(ERROR, "reset()=>%d", err);
      }
      else if (OB_SUCCESS != (err = log_buf->append_log(log_buf->get_end_pos(), start_id, end_id, buf, len)))
      {
        TBSYS_LOG(ERROR, "log_buf->append_log(pos=%ld, range=[%ld, %ld], buf=%p[%ld])=>%d",
                  log_buf->get_end_pos(), start_id, end_id, buf, len, err);
      }
      return err;
    }
  }; // end namespace updateserver
}; // end namespace oceanbase

