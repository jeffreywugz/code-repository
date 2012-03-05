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
#include "common/ob_malloc.h"
#include "common/utility.h"
#include "ob_fixed_ring_buffer.h"

using namespace oceanbase::common;

namespace oceanbase
{
  namespace updateserver
  {
    int copy_from_ring_buf(char* dest, const char* src, const int64_t ring_buf_len,
                           const int64_t start_pos, const int64_t len)
    {
      int err = OB_SUCCESS;
      int64_t bytes_to_boundary = 0;
      if (NULL == dest || NULL == src || 0 >= ring_buf_len
          || len > ring_buf_len || 0 > start_pos)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "copy_from_ring_buf(dest=%p, src=%p[%ld], range=[%ld,+%ld]): invalid argument",
                  dest, src, ring_buf_len, start_pos, len);
      }
      else
      {
        bytes_to_boundary = ring_buf_len - start_pos % ring_buf_len;
        memcpy(dest, src + start_pos % ring_buf_len, min(len, bytes_to_boundary));
        if (bytes_to_boundary < len)
        {
          memcpy(dest + bytes_to_boundary, src, len - bytes_to_boundary);
        }
      }
      return err;
    }

    int copy_to_ring_buf(char* dest, const char* src, const int64_t ring_buf_len,
                         const int64_t start_pos, const int64_t len)
    {
      int err = OB_SUCCESS;
      int64_t bytes_to_boundary = 0;
      if (NULL == dest || NULL == src || 0 >= ring_buf_len || 0 > len || len > ring_buf_len
          || 0 > start_pos)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "copy_to_ring_buf(dest=%p, src=%p[%ld], range=[%ld,+%ld]): invalid argument",
                  dest, src, ring_buf_len, start_pos, len);
      }
      else
      {
        bytes_to_boundary = ring_buf_len - start_pos % ring_buf_len;
        memcpy(dest +  start_pos % ring_buf_len, src, min(len, bytes_to_boundary));
        if (bytes_to_boundary < len)
        {
          memcpy(dest, src + bytes_to_boundary, len - bytes_to_boundary);
        }
      }
      return err;
    }

    bool is_in_range(const int64_t pos, int64_t start, int64_t end)
    {
      return pos >= start && pos < end;
    }

    bool is_point_valid_in_ring_buf(const int64_t pos, const int64_t start_pos, const int64_t end_pos, const int64_t buf_len)
    {
      return is_in_range(pos, max(start_pos, end_pos-buf_len), end_pos);
    }

    ObFixedRingBuffer::ObFixedRingBuffer() : start_pos_(0), end_pos_(0), next_end_pos_(0), start_id_(0), end_id_(0), buf_len_(0), buf_(NULL)
    {}

    ObFixedRingBuffer::~ObFixedRingBuffer()
    {}

    bool ObFixedRingBuffer::is_inited() const
    {
      return NULL != buf_ && 0 < buf_len_;
    }

    int ObFixedRingBuffer::check_state() const
    {
      return is_inited()? OB_SUCCESS: OB_NOT_INIT;
    }

    int ObFixedRingBuffer::init(const int64_t buf_len, char* buf)
    {
      int err = OB_SUCCESS;
      if (is_inited())
      {
        err = OB_INIT_TWICE;
      }
      else if (NULL == buf || 0 >= buf_len)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else
      {
        buf_ = buf;
        buf_len_ = buf_len;
      }
      return err;
    }

    int64_t ObFixedRingBuffer::get_start_pos() const
    {
      return start_pos_;
    }

    int64_t ObFixedRingBuffer::get_end_pos() const
    {
      return end_pos_;
    }

    int64_t ObFixedRingBuffer::get_next_end_pos() const
    {
      return next_end_pos_;
    }

    int ObFixedRingBuffer::reset()
    {
      int err = OB_SUCCESS;
      int64_t next_end_pos = end_pos_;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (__sync_bool_compare_and_swap(&next_end_pos_, next_end_pos, next_end_pos + 1))
      {
        err = OB_EAGAIN;
      }
      else
      {
        end_id_ = 0;
        start_id_ = 0;
        start_pos_ = next_end_pos_;
        last_reset_pos_ = next_end_pos_;
        end_pos_ = next_end_pos_;
      }
      return err;
    }

    // 允许len == 0
    int ObFixedRingBuffer::append(const int64_t pos, const int64_t start_id, const int64_t end_id,
                                  const char* buf, const int64_t len)
    {
      int err = OB_SUCCESS;
      bool granted = false;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (NULL == buf || 0 > len)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "append(buf=%p[%ld]): invalid argument", buf, len);
      }
      else if (len > buf_len_)
      {
        err = OB_BUF_NOT_ENOUGH;
        TBSYS_LOG(WARN, "append(len[%ld] > buf_len_[%ld])", len , buf_len_);
      }
      else if (pos != end_pos_ || !__sync_bool_compare_and_swap(&next_end_pos_, pos, pos + len))
      {
        err = OB_EAGAIN;
        TBSYS_LOG(DEBUG, "append(pos=%ld, next_end_pos=%ld): another appender working.", pos, next_end_pos_);
      }
      else
      {
        granted = true;
      }
      if (OB_SUCCESS == err &&
          end_id_ > 0 && end_id_ != start_id)
      {
        err = OB_DISCONTINUOUS_LOG;
        TBSYS_LOG(WARN, "log not continuous: end_id_[%ld] != start_id[%ld]", end_id_, start_id);
      }
      else if (OB_SUCCESS != (err = copy_to_ring_buf(buf_, buf, buf_len_, end_pos_, len)))
      {
        TBSYS_LOG(ERROR, "copy_to_ring_buf(ring_buf=%p[%ld]+%ld, buf=%p[%ld])=>%d", buf_, buf_len_, end_pos_, buf, len, err);
      }
      else
      {
        end_id_ = end_id;
        if (0 >= start_id_)
        {
          start_id_ = start_id;
        }
      }

      if (!granted)
      {}
      else if (OB_SUCCESS != err)
      {
        if (__sync_bool_compare_and_swap(&next_end_pos_, pos+len, end_pos_))
        {
          err = OB_ERR_UNEXPECTED;
        }
      }
      else
      {
        end_pos_ += len;
      }
      return err;
    }

    bool ObFixedRingBuffer::is_pos_valid(const int64_t pos) const
    {
      return is_point_valid_in_ring_buf(pos, start_pos_, end_pos_, buf_len_);
    }

    // 允许len == 0
    int ObFixedRingBuffer::read(const int64_t pos, char* buf, const int64_t len, int64_t& read_count) const
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (NULL == buf || 0 > len)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "read(buf=%p[%ld], pos=%ld): invalid argument", buf, len, pos);
      }
      else if (!is_pos_valid(pos))
      {
        err = OB_DATA_NOT_SERVE;
      }
      else if (0 > (read_count = min(len, end_pos_-pos)))
      {
        err = OB_ERR_UNEXPECTED; // 通过了 is_pos_valid()检查 不会进入这个分支
      }
      else if (OB_SUCCESS != (err = copy_from_ring_buf(buf, buf_, buf_len_, pos, read_count)))
      {
        TBSYS_LOG(ERROR, "copy_from_ring_buf(buf=%p[%ld], ring_buf=%p[%ld], pos=[%ld+%ld])=>%d",
                  buf, len, buf_, buf_len_, pos, read_count, err);
      }
      else if (!is_pos_valid(pos))
      {
        err = OB_DATA_NOT_SERVE;
      }
      return err;
    }
  }; // end namespace updateserver
}; // end namespace oceanbase
