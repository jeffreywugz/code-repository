/**
 * (C) 2007-2010 Alibaba Group Holding Limited.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * version 2 as published by the Free Software Foundation.
 *
 * Version: $Id$
 *
 * ob_log_buffer.cpp
 *
 * Authors:
 *   yuanqi.xhf <yuanqi.xhf@taobao.com>
 *
 */

#include "ob_data_entry_queue.h"
#include "ob_log_queue.h"

using namespace oceanbase::common;

namespace oceanbase
{
  namespace updateserver
  {
    static const int64_t LOG_FILE_ALIGN_BIT = 9;
    static bool is_aligned(int64_t x, int64_t n_bit)
    {
      return ! (x & ~((~0)<<n_bit));
    }
    int check_log_data(char* log_data, int64_t data_len, int64_t& min_id, int64_t& max_id)
    {
      int err = OB_SUCCESS;
      int64_t pos = 0;
      ObLogEntry log_entry;
      min_id = max_id = -1;
      if (NULL == log_data || data_len <= 0)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "invalid argument, log_data=%p, data_len=%ld", log_data, data_len);
      }
      while (OB_SUCCESS == err && pos < data_len)
      {
        //TBSYS_LOG(INFO, "LogTasks.check(pos=%ld, len=%ld, seq=%ld)", pos, data_len, log_entry.seq_);
        if (OB_SUCCESS != (err = log_entry.deserialize(log_data, data_len, pos)))
        {
          TBSYS_LOG(ERROR, "log_entry.deserialize(log_data=%p, data_len=%ld, pos=%ld)=>%d", log_data, data_len, pos, err);
        }
        else if (OB_SUCCESS != (err = log_entry.check_data_integrity(log_data + pos)))
        {
          TBSYS_LOG(ERROR, "log_entry.check_data_integrity()=>%d", err);
        }
        else
        {
          pos += log_entry.get_log_data_len();
          if (-1 == min_id)
          {
            min_id = log_entry.seq_;
          }
          max_id = log_entry.seq_;
        }
      }
      if (OB_SUCCESS == err && pos != data_len)
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "pos[%ld] != data_len[%ld]", pos, data_len);
      }
      return err;
    }

    int ObBatchLogEntryTask::check_data_integrity() const
    {
      int err = OB_SUCCESS;
      int64_t min_id = 0, max_id = 0;
      if (NULL == this)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "this == NULL");
      }
      else if (!is_valid())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "is_valid()=>false");
      }
      else if (for_master_ && !is_aligned(log_data_len_, LOG_FILE_ALIGN_BIT))
      {
        TBSYS_LOG(ERROR, "is_aligned(len=%ld, n_bits=%ld)=>false", log_data_len_, LOG_FILE_ALIGN_BIT);
      }
      else if (OB_SUCCESS != (err = check_log_data(log_data_, log_data_len_, min_id, max_id)))
      {
        TBSYS_LOG(ERROR, "check_log_data()=>%d", err);
      }
      else if (min_id != min_id_ || max_id_ != max_id)
      {
         err = OB_ERR_UNEXPECTED;
         TBSYS_LOG(ERROR, "tasks.id[%ld, %ld] != log_data[%p].id[%ld, %ld]", min_id_, max_id_, log_data_, min_id, max_id);
      }
      return err;
    }

    int ObBatchLogEntryTask::serialize(char* buf, int64_t len, int64_t& pos) const
    {
      int err = OB_SUCCESS;
      ObBatchLogEntryTask* that = (typeof(that))buf;
      if (!is_valid())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "is_valid()=>false");
      }
      else if (OB_SUCCESS != (err = check_data_integrity()))
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "invalid log_task");
      }
      else if (pos + get_full_data_len() > len)
      {
        err = OB_BUF_NOT_ENOUGH;
        TBSYS_LOG(ERROR, "pos[%ld] + data_len[%ld] > len[%ld]", pos, get_full_data_len(), len);
      }
      else
      {
        memcpy(buf + pos, this, sizeof(*this));
        that->log_data_ = NULL;
        that->log_tasks_ = NULL;
        pos += sizeof(*this);
        memcpy(buf + pos, log_tasks_, sizeof(*log_tasks_) * n_log_tasks_);
        pos += sizeof(*log_tasks_) * n_log_tasks_;
        memcpy(buf + pos, log_data_, log_data_len_);
        pos += log_data_len_;
      }
      return err;
    }

    int ObBatchLogEntryTask::deserialize(char* buf, int64_t len, int64_t& pos)
    {
      int err = OB_SUCCESS;
      int64_t tmp_pos = pos;
      if (NULL == this)
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "NULL == this");
      }
      else if (tmp_pos + (int64_t)sizeof(*this) > len)
      {
        err = OB_DESERIALIZE_ERROR;
        TBSYS_LOG(ERROR, "pos[%ld]+sizeof(*this)[%d] > len[%ld]", tmp_pos, sizeof(*this), len);
      }
      else
      {
        memcpy(this, buf, sizeof(*this));
      }
      if (OB_SUCCESS == err &&
          tmp_pos + get_full_data_len() > len)
      {
        err = OB_DESERIALIZE_ERROR;
        TBSYS_LOG(ERROR, "pos[%ld]+data_len[%d] > len[%ld]", tmp_pos, get_full_data_len(), len);
      }
      else
      {
        tmp_pos += sizeof(*this);
        log_tasks_ = (ObLogEntryTask*) (buf + tmp_pos);
        tmp_pos += sizeof(*log_tasks_) * n_log_tasks_;
        log_data_ = buf + tmp_pos;
        tmp_pos += log_data_len_;
        pos = tmp_pos;
      }
      return err;
    }

    ObLogIterator:: ObLogIterator()
    {
    }

    void ObLogIterator:: init(ObDataEntryIterator* iter)
    {
      data_entry_iter_ = iter;
    }

    int ObLogIterator:: commit(ObBatchLogEntryTask* log_tasks)
    {
      return data_entry_iter_->commit_by_buf((char*)log_tasks);
    }

    int ObLogIterator:: get(ObBatchLogEntryTask*& log_tasks, int64_t timeout)
    {
      int err = OB_SUCCESS;
      ObDataEntry* entry = NULL;
      ObBatchLogEntryTask* tmp_log_tasks = NULL;
      int64_t pos = 0;
      if (OB_SUCCESS != (err = data_entry_iter_->get(entry, timeout)))
      {
        if (OB_NEED_RETRY != err)
        {
          TBSYS_LOG(ERROR, "get(timeout=%ld)=>%d", timeout, err);
        }
        else
        {
          TBSYS_LOG(DEBUG, "get(timeout=%ld)=>%d", timeout, err);
        }
      }
      else
      {
        tmp_log_tasks = (typeof(tmp_log_tasks))entry->buf_;
      }

      if (OB_SUCCESS != err)
      {}
      else if(OB_SUCCESS != (err = tmp_log_tasks->deserialize(entry->buf_, entry->buf_size_, pos)))
      {
        TBSYS_LOG(ERROR, "log_tasks->deserialize()=>%d", err);
      }
      else
      {
        log_tasks = tmp_log_tasks;
      }
      return err;
    }

    int ObLogQueue:: init(int64_t block_size, int n_reader, ObLogIterator*& iters, int64_t retry_wait_time)
    {
      int err = OB_SUCCESS;
      ObDataEntryIterator* data_entry_iters = NULL;
      if (OB_SUCCESS != (err = data_entry_queue_.init(block_size, n_reader, data_entry_iters, retry_wait_time)))
      {
        TBSYS_LOG(ERROR, "data_entry_queue.init(block_size=%ld, n_reaer=%d)=%ld", block_size, n_reader);
      }
      else
      {
        for(int i = 0; i < n_reader; i++)
        {
          iters_[i].init(&data_entry_iters[i]);
        }
        iters = iters_;
      }
      return err;
    }

    int ObLogQueue:: reset()
    {
      return data_entry_queue_.clear();
    }

    int ObLogQueue:: push(const ObBatchLogEntryTask* log_tasks, int64_t timeout)
    {
      int err = OB_SUCCESS;
      ThreadSpecificBuffer::Buffer *thread_buf = thread_buffer_.get_buffer();
      ObDataBuffer log_buf(thread_buf->current(), thread_buf->remain());
      if (NULL == log_tasks)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "NULL == log_task");
      }
      else if (OB_SUCCESS != (err = log_tasks->serialize(log_buf.get_data(), log_buf.get_capacity(), log_buf.get_position())))
      {
        TBSYS_LOG(ERROR, "log_tasks->serialize()=>%d", err);
      }
      else if (OB_SUCCESS != (err = data_entry_queue_.push(log_tasks->max_id_, log_tasks->get_full_data_len(),
                                                           log_buf.get_data(), timeout)))
      {
        if (OB_NEED_RETRY != err && OB_SIZE_OVERFLOW != err)
        {
          TBSYS_LOG(ERROR, "data_entry_queue_.push(min_id=%ld, max_id=%ld, len=%ld, buf=%p)=>%d",
                    log_tasks->min_id_, log_tasks->max_id_, log_tasks->get_full_data_len(), log_tasks, err);
        }
        else
        {
          TBSYS_LOG(WARN, "data_entry_queue_.push(min_id=%ld, max_id=%ld, len=%ld, buf=%p)=>%d",
                    log_tasks->min_id_, log_tasks->max_id_, log_tasks->get_full_data_len(), log_tasks, err);
        }
      }
      if (OB_SUCCESS != err && OB_NEED_RETRY != err)
      {
        dump_for_debug();
      }
      return err;
    }

    int ObLogQueue:: enable(int64_t max_log_id)
    {
      TBSYS_LOG(INFO, "enable(max_id=%ld)", max_log_id);
      return data_entry_queue_.enable(max_log_id);
    }

    int ObLogQueue:: flush(int64_t& max_log_id)
    {
      return data_entry_queue_.flush(max_log_id);
    }

    int ObLogQueue:: set_max_log_id_can_hold(int64_t max_log_id)
    {
      data_entry_queue_.set_max_id_can_hold(max_log_id);
      return OB_SUCCESS;
    }

    int ObLogQueue:: dump_for_debug() const
    {
      return data_entry_queue_.dump_for_debug();
    }
  } // end namespace updateserver
} // end namespace oceanbase

