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
        volatile int64_t total_bytes_written_;
        volatile int64_t start_bytes_written_;
        common::ObLogCursor log_cursor_; // 记录最后的日志点
        int64_t log_buffer_len_;
        char* log_buffer_;
        int64_t n_index_entries_;
        ObBufferedLogFileIndex* indexes_;
    ObRecentLogCache::ObRecentLogCache(): total_bytes_written_(0), start_bytes_written_(0),
                                          log_cursor_(), log_buffer_len_(-1), log_buffer_(NULL),
                                          n_index_entries_(0),  indexes_(NULL)
    {}

    ObRecentLogCache::~ObRecentLogCache()
    {}

    bool ObRecentLogCache::is_inited() const
    {
      return NULL != log_buffer_ && 0 < log_buffer_len_
        && NULL != indexes_ && 0 < n_index_entries_;
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
        TBSYS_LOG(INFO, "ObRecentLogCache(indexes=%p[%ld], log_buf=%p[%ld], cursor=%s, bytes_written=[%ld, %ld])",
                  indexes_, n_index_entries_, log_buffer_, log_buffer_len_, log_cursor_.to_str(),
                  start_bytes_written_, total_bytes_written_);
        for(int i = 0; i < n_index_entries_; i++)
        {
          TBSYS_LOG(INFO, "index[%d]={file_id:%ld, bytes_written=[%ld,%ld], id:[%ld,%ld]}",
                    i, indexes_[i].file_id_, indexes_[i].bytes_written_before_,
                    indexes_[i].bytes_written_after_, indexes_[i].start_log_id_, indexes_[i].end_log_id_);
        }
      }
      return err;
    }

    int ObRecentLogCache::init(const int64_t n_index_entries, ObBufferedLogFileIndex* const indexes,
                               const int64_t log_buf_len, char* const log_buf)
    {
      int err = OB_SUCCESS;
      if (is_inited())
      {
        err = OB_INIT_TWICE;
        TBSYS_LOG(ERROR, "ObRecentLogCache:init twice.");
      }
      else if (1 >= n_index_entries || NULL == indexes || 0 >= log_buf_len || NULL == log_buf)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "ObRecentLogCache.init(index=%p[%ld], log_buf=%p[%ld])=>%d",
                  indexes, n_index_entries, log_buf, log_buf_len, err);
      }
      else
      {
        log_buffer_ = log_buf;
        log_buffer_len_ = log_buf_len;
        indexes_ = indexes;
        n_index_entries_ = n_index_entries;
      }
      return err;
    }

    int ObRecentLogCache::get_cursor(ObLogCursor& cursor) const
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else
      {
        cursor = cursor_;
      }
      return err;
    }

    int ObRecentLogCache::set_cursor(ObLogCursor& cursor)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else
      {
        cursor = cursor_;
      }
      return err;
    }

    int ObRecentLogCache::check_cursor(const ObLogCursor& cursor) const
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (cursor_.is_valid() && !cursor.equal(cursor_))
      {
        err = OB_DISCONTINUOUS_LOG;
        TBSYS_LOG(WARN, "log is not continuous[%s]", cursor.to_str());
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
        start_bytes_written_ = total_bytes_written_;
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
      else if (start + log_buffer_len_ < end || start < start_bytes_written_ || end > total_bytes_written_)
      {
        err = OB_COND_CHECK_FAIL;
        TBSYS_LOG(DEBUG, "[%ld,%ld) not in [max(%ld-%ld, %ld), %ld)",
                  start, end, total_bytes_written_, log_buffer_len_, start_bytes_written_, total_bytes_written_);
      }
      return err;
    }

    int ObRecentLogCache::fill_index(const ObLogCursor& cursor)
    {
      int err = OB_SUCCESS;
      ObBufferedLogFileIndex* tmp_index = NULL;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else
      {
        tmp_index = index_ + cursor.file_id_% n_index_entries_;
        if (tmp_index->file_id_ == cursor.file_id_)
        {
          err = OB_ENTRY_EXIST;
        }
        else
        {
          if (write_pos_ + max_log_file_size_ > log_buf_len_)
          {
            write_pos_ = 0;
          }
          tmp_index->file_id_ = -1;
          tmp_index->max_log_file_size_ = max_log_file_size_;
          tmp_index->bytes_written_before_ = bytes_written_;
          tmp_index->start_log_id_ = cursor.log_id_;
          tmp_index->end_log_id_ = cursor.log_id_;
          tmp_index->start_offset_ = write_pos_;
          tmp_index->end_offset_ = write_pos_;
          tmp_index->file_id_ = cursor.file_id_;
        }
      }
      return err;
    }

#define Index(file_id) index_[file_id% n_index_entries_]
    int ObRecentLogCache::get_index(const int64_t file_id, ObBufferedLogFileIndex& index) const
    {
      int err = OB_SUCCESS;
      uint64_t old_bytes_written_before = 0;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (Index(file_id).file_id_ != file_id
               && Index(file_id)% n_index_entries_)
      {
        err = OB_ENTRY_NOT_EXIST;
      }
      else
      {
        old_bytes_written_before = index_[file_id% n_index_entries_].bytes_written_before_;
        index = index_[file_id% n_index_entries_];
      }

      if (OB_SUCCESS == err
          && (index_[file_id% n_index_entries_].bytes_written_before_ != old_bytes_written_before
              || index_[file_id% n_index_entries_].file_id_ != file_id))
      {
        err = OB_ENTRY_NOT_EXIST;
      }
      return err;
    }

    int ObRecentLogCache::get_mutable_index(const int64_t file_id, ObBufferedLogFileIndex*& index) const
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (index_[file_id% n_index_entries_].file_id_ != file_id)
      {
        err = OB_ENTRY_NOT_EXIST;
      }
      else
      {
        index = index_ + file_id% n_index_entries_;
      }
      return err;
    }

    int ObRecentLogCache::push_log(const ObLogCursor& start_cursor, const ObLogCursor& end_cursor,
                             const char* buf, const int64_t len)
    {
      int err = OB_SUCCESS;
      ObBufferedLogFileIndex* index = NULL;
      int64_t end_pos = 0;
      ObLogCursor real_end_cursor;
      bool extra_check = true;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (!start_cursor.is_valid() || !end_cursor.is_valid() || NULL == buf || 0 >= len)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "push_log(start_cursor=%s, buf=%p, len=%ld)=>%d", start_cursor.to_str(), buf, len, err);
      }
      else if (cursor_.is_valid() && !start_cursor.equal(cursor_))
      {
        err = OB_DISCONTINUOUS_LOG;
        TBSYS_LOG(ERROR, "log is not continuous: cursor=%s", cursor_.to_str());
        TBSYS_LOG(ERROR, "log is not continuous: new_cursor=%s", start_cursor.to_str());
      }
      else if (extra_check && OB_SUCCESS != (err = parse_log_buffer(buf, len, len, end_pos, start_cursor, real_end_cursor)))
      {
        TBSYS_LOG(ERROR, "parse_log_buffer(buf[%p:%ld], start_cursor=%s)=>%d", buf, len, cursor_.to_str(), err);
      }
      else if (extra_check && !end_cursor.equal(real_end_cursor))
      {
        err = OB_INVALID_DATA;
        TBSYS_LOG(ERROR, "end_cursor.equal(");
      }
      else if (OB_SUCCESS != (err = fill_index(start_cursor)) && OB_ENTRY_EXIST != err)
      {
        TBSYS_LOG(ERROR, "fill_index()=>%d", err);
      }
      else if (OB_SUCCESS != (err = get_mutable_index(start_cursor.file_id_, index)))
      {
        TBSYS_LOG(ERROR, "get_index(file_id=%ld)=>%d", cursor_.file_id_, err);
      }
      else if (index->end_offset_ + len > log_buf_len_)
      {
        err = OB_BUF_NOT_ENOUGH;
        TBSYS_LOG(ERROR, "end_offset[%ld] + buf.len[%ld] > log_buf_len_[%ld]", index->end_offset_, len, log_buf_len_);
      }
      else
      {
        bytes_written_ += len;
        memcpy(log_buf_ + index->end_offset_, buf, len);
        index->end_offset_ += len;
        write_pos_ = index->end_offset_;
        index->end_log_id_ = end_cursor.log_id_;
        cursor_ = end_cursor;
      }
      return err;
    }

    int ObRecentLogCache::get_log(const ObLogCursor& start_cursor, ObLogCursor& end_cursor, char* buf, const int64_t len, int64_t& read_count)
    {
      return get_log_const(start_cursor, end_cursor, buf, len, read_count);
    }

    int ObRecentLogCache::get_log_const(const ObLogCursor& start_cursor, ObLogCursor& end_cursor, char* buf, const int64_t len, int64_t& read_count) const
    {
      int err = OB_SUCCESS;
      ObBufferedLogFileIndex index;
      int64_t offset = 0;
      int64_t limit = 0;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (NULL == buf || len <= 0 || !start_cursor.is_valid())
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "get_log(start_cursor=%s, buf=%p, len=%ld)=>%d", start_cursor.to_str(),
                  buf, len, err);
      }
      else if (OB_SUCCESS != (err = get_index(start_cursor.file_id_, index)))
      {
        if (OB_ENTRY_NOT_EXIST != err)
        {
          TBSYS_LOG(ERROR, "get_index(file_id=%ld)=>%d", start_cursor.file_id_, err);
        }
        else
        {
          err = OB_READ_NOTHING;
        }
      }
      else if (index.start_offset_ + start_cursor.offset_ > log_buf_len_)
      {
        err = OB_BUF_NOT_ENOUGH;
        TBSYS_LOG(ERROR, "file[%ld] offset overflow: index.offset[%ld] + start_cursor.offset[%ld] > log_buf_len[%ld]",
                  start_cursor.file_id_, index.start_offset_, start_cursor.offset_, log_buf_len_);
      }
      else if (index.end_offset_  <= index.start_offset_ + start_cursor.offset_)
      {
        err = OB_READ_NOTHING;
      }
      else
      {
        offset = index.start_offset_ + start_cursor.offset_;
        limit = min(len, index.end_offset_ - offset);
      }

      if (OB_SUCCESS != err)
      {}
      else
      {
        memcpy(buf, log_buf_ + offset, limit);
      }

      if (OB_SUCCESS != err)
      {}
      else if (index.bytes_written_before_ + log_buf_len_ - max_log_file_size_ < bytes_written_)
      {
        err = OB_READ_NOTHING;
      }
      else if (OB_SUCCESS != (err = parse_log_buffer(buf, limit, limit, read_count, start_cursor, end_cursor)))
      {
        TBSYS_LOG(ERROR, "parse_log_buffer(limit=%ld, target_cursor=%ld)=>%d", limit, start_cursor.to_str(), err);
      }
      return err;
    }

    int init_log_cache_by_alloc(ObRecentLogCache& log_cache, int64_t n_index_entries, const int64_t log_buf_len,
                                char*& cbuf)
    {
      int err = OB_SUCCESS;
      int64_t index_size = n_index_entries * sizeof(ObBufferedLogFileIndex);
      ObBufferedLogFileIndex* index = NULL;
      if (NULL == (cbuf = (char*)ob_malloc(index_size + log_buf_len)))
      {
        err = OB_ALLOCATE_MEMORY_FAILED;
        TBSYS_LOG(ERROR, "ob_malloc(%ld) failed.", index_size + log_buf_len);
      }
      else if (NULL == (index = new(cbuf)ObBufferedLogFileIndex[n_index_entries]))
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "new(%p) ObBufferedLogFileIndex[%ld] failed", cbuf, n_index_entries);
      }
      else if (OB_SUCCESS != (err = log_cache.init(n_index_entries, (ObBufferedLogFileIndex*)cbuf, log_buf_len, cbuf + index_size, max_log_file_size)))
      {
        TBSYS_LOG(ERROR, "log_cache.init(n_index=%ld, log_buf_len=%ld)=>%d", n_index_entries, log_buf_len, err);
      }
      return err;
    }

  } // end namespace updateserver
} // end namespace oceanbase
