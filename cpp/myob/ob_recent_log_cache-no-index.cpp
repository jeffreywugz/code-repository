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

#include "ob_recent_log_cache.h"
#include "ob_ups_log_utils.h"
#include "common/utility.h"

using namespace oceanbase::common;
namespace oceanbase
{
  namespace updateserver
  {
    ObRecentLogCache::ObRecentLogCache(): last_reset_pos_(0), end_pos_(0), start_pos_(0),
                                          end_log_cursor_(0), start_log_cursor_(0), log_buffer_len_(-1), log_buffer_(NULL),
                                          reserved_buf_len_(0)
    {}

    ObRecentLogCache::~ObRecentLogCache()
    {}

    bool ObRecentLogCache::is_inited() const
    {
      return NULL != log_buffer_ && 0 < log_buffer_len_ && 0 <= reserved_buf_len_;
    }

    int ObRecentLogCache::check_state() const
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      return err;
    }

    int ObRecentLogCache::dump_for_debug() const
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else
      {
        TBSYS_LOG(INFO, "ObRecentLogCache(log_buf=%p[%ld], cursor=[%ld,%ld], pos=[%ld, %ld->%ld])",
                  log_buffer_, log_buffer_len_, start_log_cursor_, end_log_cursor_, last_reset_pos_, start_pos_, end_pos_);
      }
      return err;
    }

    int ObRecentLogCache:: init(int64_t log_buf_len, char* log_buf, int64_t reserved_buf_len);
    {
      int err = OB_SUCCESS;
      if (is_inited())
      {
        err = OB_INIT_TWICE;
        TBSYS_LOG(ERROR, "ObRecentLogCache:init twice.");
      }
      else if (0 >= log_buf_len || NULL == log_buf || 0 > reserved_buf_len)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "ObRecentLogCache.init(log_buf=%p[%ld], reserved_buf_len=%ld): invalid argument",
                  log_buf, log_buf_len, reserved_buf_len);
      }
      else
      {
        log_buffer_ = log_buf;
        log_buffer_len_ = log_buf_len;
        reserved_buf_len_ = reserved_buf_len;
      }
      return err;
    }

    int ObRecentLogCache::reset()
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else
      {
        start_pos_ = end_pos_ + 1;
        last_reset_pos_ = end_pos_ + 1;
        end_pos_ ++;
        start_log_cursor_ = 0;
        end_log_cursor_ = 0;
      }
      return err;
    }

    int ObRecentLogCache::seek_(const int64_t start_pos, int64_t& real_pos, const ObLogCursor cursor) const
    {
      int err = OB_SUCCESS;
      ObLogEntry log_entry;
      char log_entry_buf[sizeof(ObLogEntry)];
      int64_t log_entry_buf_pos = 0;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else
      {
        last_reset_pos = last_reset_pos_;
        pos = start_pos;
      }
      while(OB_SUCCESS == err)
      {
        log_entry_buf_pos = 0;
        if (OB_SUCCESS != (err = copy_log_buf(last_reset_pos, pos, copy_count, log_entry_buf, sizeof(log_entry_buf))))
        {
          if (OB_DATA_NOT_SERVE != err)
          {
            TBSYS_LOG(ERROR, "copy_log_buf(last_reset_pos=%ld, pos=%ld, buf=%p[%ld])=>%d",
                      last_reset_pos, pos, log_entry_buf, sizeof(log_entry_buf), err);
          }
        }
        else if (copy_count < sizeof(log_entry_buf))
        {
          err = OB_DATA_NOT_SERVE;
        }
        else if (OB_SUCCESS != (err = ob_log_entry.deserialize(log_entry_buf, sizeof(log_entry_buf), log_entry_buf_pos)))
        {
          TBSYS_LOG(ERROR, "log_entry.deserialize(%p[%ld])=>%d", log_entry_buf, sizeof(log_entry_buf), err);
        }
        else if (cursor < ob_log_entry.seq_)
        {
          err = OB_DATA_NOT_SERVE;
        }
        else if (cursor == ob_log_entry.seq_)
        {
          break;
        }
      }
      return err;
    }

    int ObRecentLogCache::seek(const int64_t advised_pos, int64_t& real_pos, const ObLogCursor cursor) const
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (cursor > start_log_cursor_ && cursor < end_log_cursor_)
      {
        err = OB_DATA_NOT_SERVE;
      }
      else if (advised_pos >= last_reset_pos_)
      {
        // already valid
      }
      else if (OB_SUCCESS != (err = seek_(start_pos_, real_pos, cursor)))
      {
        TBSYS_LOG(ERROR, "seek_(start_pos=%ld, cursor=%ld)=>%d", start_pos_, cursor, err);
      }
      return err;
    }

    int64_t ring_buf_pos(const int64_t len, const int64_t pos)
    {
      return (pos - 1) % len + 1;
    }

    int copy_from_ring_buf(char* dest, const char* src, const int64_t ring_buf_len,
                           const int64_t start_pos, const int64_t end_pos)
    {
      int err = OB_SUCCESS;
      if (NULL == dest || NULL == src || 0 >= ring_buf_len || start_pos > end_pos
          || start_pos + ring_buf_len < end_pos || 0 > start_pos)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "copy_from_ring_buf(dest=%p, src=%p[%ld], range=[%ld,%ld]): invalid argument",
                  dest, src, ring_buf_len, start_pos, end_pos);
      }
      else
      {
        buf_boundary = ((start_pos + ring_buf_len - 1)/ ring_buf_len) * ring_buf_len;
        memcpy(dest, src, start_pos % ring_buf_len, ring_buf_pos(ring_buf_len, min(buf_boundary, end_pos)));
        if (buf_boundary < end_pos)
        {
          memcpy(dest, src, 0, ring_buf_pos(ring_buf_len, end_pos));
        }
      }
      return err;
    }

    int ObRecentLogCache::copy_log_buf(const int64_t last_reset_pos, const int64_t pos, uint64_t& copy_count,
                                       char* buf, const int64_t len) const
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (NULL == buf || 0 >= len || 0 > pos || 0 > last_reset_pos)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "copy_log_buf(buf=%p[%ld], last_reset_pos=%ld, pos=%ld): invalid argument",
                  buf, len, last_reset_pos, pos);
      }
      else
      {
        end_pos = min(end_pos_, pos + len);
      }

      if (OB_SUCCESS != err)
      {}
      else if (pos >= end_pos)
      {
        err = OB_DATA_NOT_SERVE;
      }
      else if (OB_SUCCESS != (err = copy_from_ring_buf(buf, log_buffer_, log_buffer_len_, pos, end_pos)))
      {
        TBSYS_LOG(ERROR, "copy_from_ring_buf(buf=%p[%ld], log_buf=%p[%ld], pos=[%ld,%ld])=>%d",
                  buf, len, log_buffer_, log_buffer_len_, pos, end_pos, err);
      }
      else if (OB_SUCCESS != (err = check_range_validity(pos, end_pos)))
      {
        if (OB_COND_CHECK_FAIL != err)
        {
          TBSYS_LOG(ERROR, "check_range_validity([%ld,%ld])=>%d", pos, end_pos, err);
        }
        else
        {
          err = OB_DATA_NOT_SERVE;
        }
      }
      else if (last_reset_pos != last_reset_pos_)
      {
        err = OB_DATA_NOT_SERVE;
      }
      else
      {
        copy_count = end_pos - pos;
      }
      return err;
    }

    int ObRecentLogCache::check_range_validity(const uint64_t start, const uint64_t end) const
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (start > end)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "start[%ld] > end[%ld]", start, end);
      }
      else if (start + log_buffer_len_ < end || start < start_pos_ || end > end_pos_)
      {
        err = OB_COND_CHECK_FAIL;
        TBSYS_LOG(DEBUG, "[%ld,%ld) not in [%ld, %ld)",
                  start, end, start_pos_, end_pos_);
      }
      return err;
    }

    int ObRecentLogCache::push_log(const ObLogCursor& start_cursor, const ObLogCursor& end_cursor,
                             const char* buf, const int64_t len)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (NULL == buf || 0 >= len || 0 >= start_cursor || end_cursor < start_cursor)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "push_log(cursor=[%ld,%ld], buf=%p[%ld]): invalid argument", start_cursor, end_cursor, buf, len);
      }
      else if (end_log_cursor_ > 0 && end_log_cursor_ != start_cursor
               && OB_SUCCESS != (err = reset()))
      {
        TBSYS_LOG(ERROR, "reset()=>%d", err);
      }
      else if (OB_SUCCESS != (err = reclaim_buffer(len + reserved_buf_len_)))
      {
        TBSYS_LOG(ERROR, "reclaim_buffer(len[%ld] + reserved_buf_len_[%ld])=>%d",
                  len, reserved_buf_len_, err);
      }
      else
      {
        memcpy(log_buffer_ + end_pos_, buf, len);
        end_pos_ + =len;
        end_log_cursor_ = end_cursor;
        if (start_log_cursor_ <= 0)
        {
          start_log_cursor_ = start_cursor_;
        }
      }
      return err;
    }

    int ObRecentLogCache::get_log(const int64_t advised_pos, int64_t& real_pos, const ObLogCursor& start_cursor, ObLogCursor& end_cursor, char* buf, const int64_t len, int64_t& read_count) const
    {
      int err = OB_SUCCESS;
      int64_t copy_count = 0;
      int64_t old_last_reset_pos = 0;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (NULL == buf || len <= 0 || 0 >= start_cursor)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "get_log(start_cursor=%ld, buf=%p[%ld]): invalid argument", start_cursor, buf, len);
      }
      else
      {
        old_last_reset_pos = last_reset_pos_;
      }

      if (OB_SUCCESS != err)
      {}
      else if (OB_SUCCESS != (err = seek(advised_pos, real_pos, start_cursor)))
      {
        if (OB_DATA_NOT_SERVE != err)
        {
          TBSYS_LOG(ERROR, "seek(pos=%ld, cursor=%ld)=>%d", advised_pos, start_cursor);
        }
        else
        {
          err = OB_READ_NOTHING;
        }
      }
      else if (OB_SUCCESS != (err = copy_to_buf(buf, len, pos, copy_count)))
      {
        if (OB_DATA_NOT_SERVE != err)
        {
          TBSYS_LOG(ERROR, "copy_to_buf(%p[%ld], read_count=%ld)=>%d", buf, len, read_count, err);
        }
        else
        {
          err = OB_READ_NOTHING;
        }
      }
      else if (old_last_reset_pos != last_reset_pos_)
      {
        err = OB_READ_NOTHING;
      }
      else if (OB_SUCCESS != (err = trim_log_buffer(buf, copy_count, read_count, start_cursor, end_cursor)))
      {
        TBSYS_LOG(ERROR, "parse_log_buffer(buf=%p[%ld], start_cursor=%ld)=>%d",
                  buf, copy_count, start_cursor, err);
      }
      return err;
    }

    int init_log_cache_by_alloc(ObRecentLogCache& log_cache, const int64_t log_buf_len,
                                const int64_t reserved_buf_len, char*& cbuf)
    {
      int err = OB_SUCCESS;
      if (NULL == (cbuf = (char*)ob_malloc(log_buf_len)))
      {
        err = OB_ALLOCATE_MEMORY_FAILED;
        TBSYS_LOG(ERROR, "ob_malloc(%ld) failed.", log_buf_len);
      }
      else if (OB_SUCCESS != (err = log_cache.init(log_buf_len, cbuf, reserved_buf_len)))
      {
        TBSYS_LOG(ERROR, "log_cache.init(log_buf=%p[%ld])=>%d", cbuf, log_buf_len, err);
      }
      return err;
    }
  } // end namespace updateserver
} // end namespace oceanbase
