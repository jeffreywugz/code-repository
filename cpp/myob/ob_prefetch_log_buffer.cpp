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
#include "tbsys.h"
#include "ob_prefetch_log_buffer.h"
#include "ob_ups_log_utils.h"

using namespace oceanbase::common;

namespace oceanbase
{
  namespace updateserver
  {
    ObPrefetchLogBuffer::ObPrefetchLogBuffer(): read_pos_(0)
    {}

    ObPrefetchLogBuffer::~ObPrefetchLogBuffer()
    {}

    int ObPrefetchLogBuffer::reset(const int64_t end_id)
    {
      int err = OB_SUCCESS;
      int64_t next_end_pos = end_pos_;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (!__sync_bool_compare_and_swap(&next_end_pos_, next_end_pos, next_end_pos + 1))
      {
        err = OB_EAGAIN;
        TBSYS_LOG(INFO, "reset(end_id=%ld, end_pos=%ld): EAGAIN", end_id, end_pos_);
      }
      else
      {
        end_id_ = end_id;
        read_pos_ = next_end_pos_;
        end_pos_ = next_end_pos_;
        TBSYS_LOG(INFO, "reset(end_id=%ld, end_pos=%ld): OK.", end_id, end_pos_);
      }
      return err;
    }

    int ObPrefetchLogBuffer::dump_for_debug() const
    {
      int err = OB_SUCCESS;
      TBSYS_LOG(ERROR, "buf_size=%ld, pos=[%ld, %ld), end_id=%ld", buf_len_, read_pos_, end_pos_, end_id_);
      return err;
    }

    int ObPrefetchLogBuffer::get_log(const int64_t start_id, int64_t& end_id,
                                    char* buf, const int64_t len, int64_t& read_count)
    {
      int err = OB_SUCCESS;
      int64_t copy_count = 0;
      int64_t real_start_id = start_id;

      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (NULL == buf || 0 >= len || 0 >= start_id)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else if (OB_SUCCESS != (err = read(read_pos_, buf, len, copy_count)))
      {
        if (OB_DATA_NOT_SERVE != err)
        {
          TBSYS_LOG(ERROR, "log_buf_.append_log(pos, buf, len)=>%d", err);
        }
      }
      else if (OB_SUCCESS != (err = trim_log_buffer(buf, copy_count, read_count, real_start_id, end_id)))
      {
        err = OB_DISCONTINUOUS_LOG;
        TBSYS_LOG(WARN, "trim_log_buffer(buf=%p[%ld], start_id=%ld)=>%d", buf, copy_count, start_id);
      }
      else if (real_start_id != start_id)
      {
        err = OB_DISCONTINUOUS_LOG;
        TBSYS_LOG(WARN, "real_start_id[%ld] != start_id[%ld]", real_start_id, start_id);
      }
      else
      {
        read_pos_ += copy_count;
      }
      return err;
    }

    int ObPrefetchLogBuffer::append_log(const int64_t pos, const int64_t start_id, const int64_t end_id,
                                        const char* buf, const int64_t len)
    {
      int err = OB_SUCCESS;
      bool granted = false;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (NULL == buf || 0 > len || 0 >= start_id || end_id < start_id)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "append(buf=%p[%ld]): invalid argument", buf, len);
      }
      else if (len > buf_len_)
      {
        err = OB_BUF_NOT_ENOUGH;
        TBSYS_LOG(WARN, "append(len[%ld] > buf_len_[%ld])", len , buf_len_);
      }
      else if (read_pos_ + buf_len_ <= pos + len) // 检查是否会覆盖未读的日志
      {
        err = OB_EAGAIN;
      }
      else if (pos != end_pos_ || !(granted = __sync_bool_compare_and_swap(&next_end_pos_, pos, pos + len)))
      {
        err = OB_EAGAIN;
        TBSYS_LOG(DEBUG, "append(pos=%ld, next_end_pos=%ld): another appender working.", pos, next_end_pos_);
      }
      else if (end_id_ > 0 && end_id_ != start_id)
      {
        err = OB_DISCONTINUOUS_LOG;
        TBSYS_LOG(WARN, "end_id_[%ld] != start_id[%ld]", end_id_, start_id);
      }
      else if (OB_SUCCESS != (err = append(pos, buf, len)))
      {
        TBSYS_LOG(ERROR, "append(pos=%ld, buf=%p[%ld])=>%d", pos, buf, len, err);
      }
      else
      {
        end_id_ = end_id;
      }

      if (granted && OB_SUCCESS != err
          && !__sync_bool_compare_and_swap(&next_end_pos_, pos+len, end_pos_))
      {
        err = OB_ERR_UNEXPECTED;
      }
      return err;
    }
  }; // end namespace updateserver
}; // end namespace oceanbase

