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

#ifndef __OB_COMMON_OB_PIPE_BUFFER_H__
#define __OB_COMMON_OB_PIPE_BUFFER_H__
#include "ob_pipe_buffer.h"
#include "utility.h"

namespace oceanbase
{
  namespace common
  {
    bool ObPipeBuffer::is_inited()
    {
      return NULL != buf_ && capacity_ != 0;
    }

    int ObPipeBuffer::init(int64_t capacity, char* buf)
    {
      int err = OB_SUCCESS;
      if (0 == capacity || !is2n(capacity) || NULL == buf)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "init(capacity=%ld, buf=%p):INVALID_ARGUMENT", capacity, buf);
      }
      else if (0 != capacity_ || NULL != buf_)
      {
        err = OB_INIT_TWICE;
      }
      else
      {
        buf_ = buf;
        capacity_ = capacity;
      }
      return err;
    }

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

    int ObPipeBuffer::push(char* buf, const int64_t len)
    {
      int err = OB_SUCCESS;
      else if (push_ + len > pop_ + capacity_)
      {
        err = OB_EAGAIN;
      }
      else if (OB_SUCCESS != (err = copy_to_ring_buf(buf_, buf, capacity_, push_, len)))
      {
        TBSYS_LOG(ERROR, "copy_to_ring_buf(buf_=%p[%ld], buf=%p[%ld], pos=[%ld,%ld])=>%d",
                  buf_, capacity_, buf, len, pop_, push_, err);
      }
      return err;
    }

    int ObPipeBuffer::pop(char* buf, const int64_t limit, int64_t& read_count)
    {
      int err = OB_SUCCESS;
      int64_t bytes2read = 0;
      read_count = 0;
      if (0 >= (bytes2read = push_ - pop_))
      {
        err = OB_EAGAIN;
      }
      else if (OB_SUCCESS != (err = copy_from_ring_buf(buf, buf_, capacity_, pop_, bytes2read)))
      {
        TBSYS_LOG(ERROR, "copy_from_ring_buf(buf=%p[%ld], buf_=%p[%ld], pos=[%ld,%ld])=>%d",
                  buf, bytes2read, buf_, capacity_, pop_, push_, err);
      }
      else
      {
        read_count = bytes2read;
      }
      return err;
    }

    static int64_t calc_end_time(const int64_t timeout_us)
    {
      return timeout_us > 0? tbsys::CTimeUtil::getTime() + timeout_us: -1;
    }

    static int64_t calc_remain_time(const int64_t end_time_us, const int64_t default_wait_time_us)
    {
      return end_time_us > 0? end_time_us - tbsys::CTimeUtil::getTime(): default_wait_time_us;
    }

    static inline bool is_timeout(const int64_t end_time)
    {
      return end_time > 0 && tbsys::CTimeUtil::getTime() > end_time;
    }

    int ObPipeBuffer::push(char* buf, const int64_t len, const int64_t timeout_us)
    {
      int err = OB_EAGAIN;
      int64_t end_time_us = calc_end_time(timeout_us);
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (NULL == buf || 0 > len || len > capacity_)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "buf=%p[%ld]", buf, len);
      }
      else
      {
        while(OB_EAGAIN == (err = push(buf, len)) && !is_timeout(end_time_us))
          ;
      }
      if (OB_SUCCESS == err)
      {
        cond_.lock();
        cond_.signal();
        cond_.unlock();
      }
      return err;
    }

    int ObPipeBuffer::pop(char* buf, const int64_t limit, int64_t& read_count, const int64_t timeout_us)
    {
      int err = OB_EAGAIN;
      int64_t end_time_us = calc_end_time(timeout_us);
      int64_t default_wait_time_us = 10 * 1000;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (NULL == buf || 0 > limit)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "buf=%p[%ld]", buf, limit);
      }
      cond_.lock();
      while(OB_EAGAIN == err
            && 0 < (remain_time_us = calc_remain_time(end_time_us, default_wait_time_us)))
      {
        if (push_ > pop_)
        {
          err = OB_SUCCESS;
        }
        else
        {
          cond_.wait(remain_time_us);
        }
      }
      cond_.unlock();
      if (OB_SUCCESS == err)
      {
        err = pop(buf, limit, read_count);
      }
      return err;
    }

    int ObPipeBuffer::consumed(char* buf, const int64_t len)
    {
      int err = OB_SUCCESS;
      if (NULL == buf || 0 > len)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "consumed(buf=%p[%ld]):INVALID_ARGUMENT", buf, len);
      }
      else
      {
        __sync_fetch_and_add(&pop_, len);
      }
      return err;
    }
  }; // end namespace common
}; // end namespace oceanbase
#endif /* __OB_COMMON_OB_PIPE_BUFFER_H__ */
