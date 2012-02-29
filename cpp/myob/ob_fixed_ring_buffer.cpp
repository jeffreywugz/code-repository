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

    ObFixedRingBuffer::ObFixedRingBuffer() : end_pos_(0), buf_len_(0), buf_(NULL)
    {}

    ObFixedRingBuffer::~ObFixedRingBuffer()
    {
      if (NULL != buf_)
      {
        ob_free(buf_);
        buf_ = NULL;
      }
    }

    bool ObFixedRingBuffer::is_inited() const
    {
      return NULL != buf_;
    }

    int ObFixedRingBuffer::init(int64_t buf_len)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = is_inited()))
      {
        err = OB_INIT_TWICE;
      }
      else if (0 > buf_len)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else if (NULL == (buf_ = (char*)ob_malloc(buf_len)))
      {
        err = OB_ALLOCATE_MEMORY_FAILED;
        TBSYS_LOG(ERROR, "ob_malloc(%ld) failed.", buf_len);
      }
      else
      {
        buf_len_ = buf_len;
      }
      return err;
    }

    int64_t ObFixedRingBuffer::get_end_pos() const
    {
      return end_pos_;
    }

    int64_t ObFixedRingBuffer::get_next_end_pos() const
    {
      return next_end_pos_;
    }

    int ObFixedRingBuffer::append(const int64_t pos, const char* buf, const int64_t len)
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
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
      else if (pos != end_pos_ || __sync_bool_compare_and_swap(&next_end_pos_, pos + len, -1))
      {
        err = OB_EAGAIN;
        TBSYS_LOG(DEBUG, "append(pos=%ld, next_end_pos=%ld): another appender working.", pos, next_end_pos_);
      }
      else if (OB_SUCCESS != (err = copy_to_ring_buf(buf_, buf, buf_len_, end_pos_, len)))
      {
        TBSYS_LOG(ERROR, "copy_to_ring_buf(ring_buf=%p[%ld]+%ld, buf=%p[%ld])=>%d", buf_, buf_len_, end_pos_, buf, len, err);
      }
      else
      {
        end_pos_ += len;
      }
      return err;
    }

    int ObFixedRingBuffer::read(const int64_t pos, char* buf, const int64_t len, int64_t& read_count)
    {
      int err = OB_SUCCESS;
      int64_t end_pos = 0;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (NULL == buf || 0 > len)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "read(buf=%p[%ld], pos=%ld): invalid argument", buf, len, pos);
      }
      else
      {
        end_pos = min(end_pos_, pos + len);
      }

      if (OB_SUCCESS != err)
      {}
      else if (pos > end_pos)
      {
        err = OB_DATA_NOT_SERVE;
      }
      else if (OB_SUCCESS != (err = copy_from_ring_buf(buf, buf_, buf_len_, pos, end_pos - pos)))
      {
        TBSYS_LOG(ERROR, "copy_from_ring_buf(buf=%p[%ld], ring_buf=%p[%ld], pos=[%ld,%ld])=>%d",
                  buf, len, buf_, buf_len_, pos, end_pos, err);
      }
      else
      {
        read_count = end_pos - pos;
      }
      return err;
    }
  }; // end namespace updateserver
}; // end namespace oceanbase
