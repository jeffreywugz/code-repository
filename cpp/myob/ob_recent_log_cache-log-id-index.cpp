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
#include "common/utility.h"
#include "common/ob_malloc.h"
#include "ob_ups_log_utils.h"

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

    ObRecentLogCache::ObRecentLogCache(): is_inited_(false), end_id_(0)
    {}

    ObRecentLogCache::~ObRecentLogCache()
    {}

    bool ObRecentLogCache::is_inited() const
    {
      return is_inited_;
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

    int ObRecentLogCache:: init(int64_t log_buf_len, int64_t n_indexes)
    {
      int err = OB_SUCCESS;
      if (is_inited())
      {
        err = OB_INIT_TWICE;
        TBSYS_LOG(ERROR, "ObRecentLogCache:init twice.");
      }
      else if (0 >= log_buf_len || 0 >= n_indexes)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "ObRecentLogCache.init(log_buf_len=[%ld], n_indexes=%ld): invalid argument",
                  log_buf_len, n_indexes);
      }
      else if (OB_SUCCESS != (err = log_buf_.init(log_buf_len)))
      {
        TBSYS_LOG(ERROR, "log_buf_.init(log_buf_len=%ld)=>%d", log_buf_len, err);
      }
      else if (OB_SUCCESS != (err = log_pos_index_.init(n_indexes)))
      {
        TBSYS_LOG(ERROR, "log_pos_index_.init(n_indexes=%ld)=>%d", n_indexes, err);
      }
      else
      {
        version_ = 1;
        is_inited_ = true;
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
        version_++;
      }
      return err;
    }

    int ObRecentLogCache::push_log(const int64_t start_id, const int64_t end_id,
                             const char* buf, const int64_t len)
    {
      int err = OB_SUCCESS;
      int64_t real_start_id = 0;
      int64_t real_end_id = 0;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (NULL == buf || 0 > len || 0 >= start_id || start_id > end_id)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "push_log(range=[%ld,%ld], buf=%p[%ld]): invalid argument", start_id, end_id, buf, len);
      }
      else if (end_id_ > 0 && end_id_ != start_id)
      {
        err = OB_DISCONTINUOUS_LOG;
      }
      else if (OB_SUCCESS != (err = update_log_index_in_buf(version_, log_pos_index_, log_buf_.get_end_pos(),
                                                            real_start_id, real_end_id, buf, len)))
      {
        TBSYS_LOG(ERROR, "update_index(log_range=[%ld,%ld], buf=%p[%ld])=>%d", start_id, end_id, buf, len, err);
      }
      else if (len > 0 && (start_id != real_start_id || end_id != real_end_id))
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "range[%ld,%ld] != real_range[%ld,%ld]", start_id, end_id, real_start_id, real_end_id);
      }
      else if (OB_SUCCESS != (err = log_buf_.append(log_buf_.get_end_pos(), buf, len)))
      {
        TBSYS_LOG(ERROR, "copy_to_ring_buf(ring_buf=%p, buf=%p[%ld])=>%d", &log_buf_, buf, len, err);
      }
      else
      {
        end_id_ = end_id;
      }
      return err;
    }

    int ObRecentLogCache::get_log(const int64_t start_id, int64_t& end_id, char* buf, const int64_t len, int64_t& read_count)
    {
      int err = OB_SUCCESS;
      int64_t old_version = version_;
      int64_t copy_count = 0;
      int64_t pos = 0;
      int64_t real_start_id = 0;
      int64_t real_end_id = 0;
      end_id = start_id;
      read_count = 0;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (NULL == buf || len <= 0 || 0 >= start_id)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "get_log(start_log_id=%ld, buf=%p[%ld]): invalid argument", start_id, buf, len);
      }
      else if (OB_SUCCESS != (err = log_pos_index_.get(old_version, start_id, pos)))
      {
        if (OB_ENTRY_NOT_EXIST != err)
        {
          TBSYS_LOG(ERROR, "log_pos_index_.get(version=%ld, start_id=%ld)=>%d", old_version, start_id, err);
        }
        else
        {
          err = OB_SUCCESS;
        }
      }
      else if (OB_SUCCESS != (err = log_buf_.read(pos, buf, len, copy_count)))
      {
        TBSYS_LOG(ERROR, "log_buf.read(buf=%p[%ld], pos=%ld)=>%d", buf, len, pos, err);
      }
      else if (old_version != version_)
      {}
      else if (OB_SUCCESS != (err = trim_log_buffer(buf, copy_count, read_count, real_start_id, real_end_id)))
      {
        TBSYS_LOG(ERROR, "parse_log_buffer(buf=%p[%ld], start_id=%ld)=>%d",
                  buf, copy_count, start_id, err);
      }
      else if (read_count <= 0)
      {} // read nothing
      else if (start_id != real_start_id)
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "get_log(start_id[%ld] != read_start_id[%ld])", start_id, real_start_id);
      }
      else
      {
        end_id = real_end_id;
      }
      return err;
    }
  } // end namespace updateserver
} // end namespace oceanbase
