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
#include "ob_log_replay_worker.h"

namespace oceanbase
{
  namespace updateserver
  {
    ObLogReplayWorker::ObLogReplayWorker(): n_worker_(0), log_applier_(NULL), is_log_buf_free_(true),
                                            replay_cursor_(), trans_mgr_(NULL),
                                            next_submit_log_id_(0), next_commit_log_id_(0), last_barrier_log_id_(0)
    {}

    ObLogReplayWorker::~ObLogReplayWorker()
    {}

    int ObLogReplayWorker::init(TransMgr* trans_mgr, IAsyncLogApplier* log_applier, const int64_t n_worker)
    {
      int err = OB_SUCCESS;
      if (NULL == trans_mgr || NULL == log_applier || n_worker <= 0 || n_worker > MAX_N_WORKER)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else if (NULL != trans_mgr_ || NULL != log_applier_ || n_worker_ > 0)
      {
        err = OB_INIT_TWICE;
      }
      else
      {
        trans_mgr_ = trans_mgr;
        log_applier_ = log_applier;
        n_worker_ = n_worker;
        setThreadCount(n_worker_);
        TBSYS_LOG(INFO, "log_replay_worker.init(log_applier=%p, n_worker=%ld): success", log_applier, n_worker);
      }
      return err;
    }

    bool ObLogReplayWorker::is_inited() const
    {
      return NULL != trans_mgr_ && NULL != log_applier_ && n_worker_ >0 ;
    }
      
    int ObLogReplayWorker::get_replay_cursor(ObLogCursor& cursor) const
    {
      int err = OB_SUCCESS;
      cursor = replay_cursor_;
      return err;
    }

    int ObLogReplayWorker::update_replay_cursor(const ObLogCursor& cursor)
    {
      int err = OB_SUCCESS;
      if (replay_cursor_.newer_than(cursor))
      {
        err = OB_DISCONTINUOUS_LOG;
        TBSYS_LOG(ERROR, "update_replay_cursor(replay_cursor[%ld:%ld+%ld] > curosr[%ld:%ld+%ld])",
                  replay_cursor_.log_id_, replay_cursor_.file_id_, replay_cursor_.offset_,
                  cursor.log_id_, cursor.file_id_, cursor.offset_);
      }
      else
      {
        replay_cursor_ = cursor;
      }
      return err;
    }

    int ObLogReplayWorker::start_log(const ObLogCursor& log_cursor)
    {
      int err = OB_SUCCESS;
      if (!log_cursor.is_valid())
      {
        err = OB_INVALID_ARGUMENT;
      }
      else if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (replay_cursor_.is_valid())
      {
        if (log_cursor.equal(replay_cursor_))
        {
          TBSYS_LOG(WARN, "start_log(log_cursor=%s): ALREADY STARTED.", log_cursor.to_str());
        }
        else
        {
          err = OB_INIT_TWICE;
        }
      }
      else
      {
        replay_cursor_ = log_cursor;
      }
      return err;
    }

    void ObLogReplayWorker::run(tbsys::CThread* thread, void* arg)
    {
      int err = OB_SUCCESS;
      int64_t id = (int64_t)arg;
      UNUSED(thread);
      TBSYS_LOG(INFO, "log replay worker[%ld] start", id);
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (OB_SUCCESS != (err = do_work(id)))
      {
        stop();
        TBSYS_LOG(ERROR, "do_work(id=%ld)=>%d", id, err);
      }
      TBSYS_LOG(INFO, "log replay worker[%ld] exit: err=%d", id, err);
    }

    int ObLogReplayWorker::do_work(int64_t thread_id)
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (thread_id < 0 || thread_id >= n_worker_)
      {
        err = OB_INVALID_ARGUMENT;
      }
      while (!_stop && OB_SUCCESS == err)
      {
        if (OB_SUCCESS != (err = replay(worker_state_[thread_id].task_))
            && OB_EAGAIN != err)
        {
          TBSYS_LOG(ERROR, "replay(thread_id=%ld)=>%d", thread_id, err);
        }
        else if (OB_EAGAIN == err)
        {
          err = OB_SUCCESS;
        }
      }
      return err;
    }

    bool ObLogReplayWorker::is_task_finished(const int64_t log_id) const
    {
      return log_id < next_commit_log_id_;
    }

    int ObLogReplayWorker::wait_task(int64_t id) const
    {
      int err = OB_SUCCESS;
      while(!is_task_finished(id))
        ;
      return err;
    }

    int ObLogReplayWorker::replay(ObLogTask& task)
    {
      int err = OB_SUCCESS;
      //TBSYS_LOG(INFO, "replay(task.log_id[%ld], next_submit_log_id[%ld], next_commit_log_id[%ld])", task.log_id_, next_submit_log_id_, next_commit_log_id_);
      if (task.log_id_ == 0 || task.log_id_ < next_commit_log_id_)
      {
        err = OB_EAGAIN;
      }
      else if (!is_task_finished(task.barrier_log_id_))
      {
        err = OB_EAGAIN;
      }
      else if (OB_SUCCESS != (err = log_applier_->start_transaction(task)))
      {
        TBSYS_LOG(ERROR, "log_applier->start_transaction()=>%d", err);
      }
      else if (OB_SUCCESS != (err = log_applier_->apply(task)))
      {
        TBSYS_LOG(ERROR, "log_applier->apply()=>%d", err);
      }
      else if (OB_SUCCESS != (err = wait_task(task.log_id_ - 1)))
      {
        TBSYS_LOG(ERROR, "wait_task()=>%d", err);
      }
      else if (OB_SUCCESS != (err = log_applier_->end_transaction(task)))
      {
        TBSYS_LOG(ERROR, "log_applier->end_transaction()=>%d", err);
      }
      else
      {
        next_commit_log_id_  = task.log_id_ + 1;
      }
      return err;
    }
    
    int ObLogReplayWorker::alloc(char*& buf, const int64_t len)
    {
      int err = OB_SUCCESS;
      if (!is_log_buf_free_ || len > MAX_LOG_BUF_SIZE)
      {
        err = OB_MEM_OVERFLOW;
        TBSYS_LOG(ERROR, "alloc(is_log_buf_free=%s, len=%ld): FAIL", STR_BOOL(is_log_buf_free_), len);
      }
      else
      {
        buf = log_buf_;
        is_log_buf_free_ = false;
      }
      return err;
    }

    int ObLogReplayWorker::release(char* buf, const int64_t len)
    {
      int err = OB_SUCCESS;
      UNUSED(len);
      if (buf != log_buf_)
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "buf[%p] != log_buf_[%p]", buf, log_buf_);
      }
      else
      {
        is_log_buf_free_ = true;
      }
      return err;
    }

    static int get_first_log_id(int64_t& log_id, const char* buf, const int64_t len, const int64_t pos)
    {
      int err = OB_SUCCESS;
      ObLogEntry entry;
      int64_t new_pos = pos;
      if (new_pos + entry.get_serialize_size() > len)
      {
        err = OB_PARTIAL_LOG;
      }
      else if (OB_SUCCESS != (err = entry.deserialize(buf, len, new_pos)))
      {
        TBSYS_LOG(ERROR, "entry.deserialize(buf=%p[%ld], pos=%ld)=>%d", buf, len, new_pos, err);
      }
      else
      {
        log_id = entry.seq_;
      }
      return err;
    }

    static int is_barrier_log(bool& is_barrier, const LogCommand cmd, const char* buf, const int64_t len)
    {
      int err = OB_SUCCESS;
      is_barrier = true;
      if (NULL == buf || 0 >= len)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else if (OB_LOG_NOP == cmd)
      {
        is_barrier = false;
      }
      else if (OB_LOG_UPS_MUTATOR == cmd)
      {
        ObUpsMutator mutator;
        int64_t pos = 0;
        if (OB_SUCCESS != (err = mutator.deserialize(buf, len, pos)))
        {
          TBSYS_LOG(ERROR, "mutator.deserialize(buf=%p[%ld])=>%d", buf, len, err);
        }
        else if (mutator.is_normal_mutator())
        {
          is_barrier = false;
        }
      }
      else
      {
        is_barrier = true;
      }
      return err;
    }

    int64_t ObLogReplayWorker::get_replay_log_id() const
    {
      return replay_cursor_.log_id_;
    }

    int ObLogReplayWorker::submit(ObLogTask& task, int64_t& log_id, const char* buf, int64_t len, int64_t& pos, const ReplayType replay_type)
    {
      int err = OB_SUCCESS;
      int64_t new_pos = pos;
      bool check_integrity = true;
      bool is_barrier = true;
      //TBSYS_LOG(INFO, "submit(task.log_id[%ld], next_submit_log_id[%ld], next_commit_log_id[%ld])", task.log_id_, next_submit_log_id_, next_commit_log_id_);
      if (_stop)
      {
        err = OB_CANCELED;
      }
      else if (task.log_id_ != 0 && task.log_id_ >= next_commit_log_id_)
      {
        err = OB_EAGAIN;
        //TBSYS_LOG(INFO, "submit(task.log_id[%ld] >= next_commit_log_id[%ld])", task.log_id_, next_commit_log_id_);
      }
      else if (new_pos + task.log_entry_.get_serialize_size() > len)
      {
        err = OB_PARTIAL_LOG;
      }
      else if (OB_SUCCESS != (err = task.log_entry_.deserialize(buf, len, new_pos)))
      {
        TBSYS_LOG(ERROR, "task.log_entry.deserialize()=>%d", err);
      }
      else if (new_pos + task.log_entry_.get_log_data_len() > len)
      {
        err = OB_PARTIAL_LOG;
      }
      else if (check_integrity && OB_SUCCESS != (err = task.log_entry_.check_header_integrity()))
      {
        TBSYS_LOG(ERROR, "log_entry.check_header_integrity()=>%d", err);
      }
      else if (check_integrity && OB_SUCCESS != (err = task.log_entry_.check_data_integrity(buf + new_pos)))
      {
        TBSYS_LOG(ERROR, "log_entry.check_data_integrity()=>%d", err);
      }
      else if (next_submit_log_id_ != (int64_t)task.log_entry_.seq_)
      {
        err = OB_DISCONTINUOUS_LOG;
        TBSYS_LOG(ERROR, "next_submit_log_id[%ld] != task.log_id[%ld]", next_submit_log_id_, task.log_entry_.seq_);
      }
      else if (OB_SUCCESS != (err = is_barrier_log(is_barrier, (LogCommand)task.log_entry_.cmd_,
                                                   buf + new_pos, task.log_entry_.get_log_data_len())))
      {
        TBSYS_LOG(ERROR, "is_barrier_log()=>%d", err);
      }
      else if (OB_SUCCESS != (err = replay_cursor_.advance(task.log_entry_)))
      {
        TBSYS_LOG(ERROR, "replay_cursor.advance()=>%d", err);
      }
      else
      {
        if (is_barrier)
        {
          last_barrier_log_id_ = task.log_entry_.seq_;
          task.barrier_log_id_ = last_barrier_log_id_ - 1;
        }
        else
        {
          task.barrier_log_id_ = last_barrier_log_id_;
        }
        pos = new_pos + task.log_entry_.get_log_data_len();
        task.commit_id_ = trans_mgr_->generate_commit_id();
        task.log_data_ = buf + new_pos;
        task.replay_type_ = replay_type;
        task.log_id_ = task.log_entry_.seq_;
        next_submit_log_id_ = task.log_id_ + 1;
        log_id = task.log_id_;
      }
      return err;
    }

    int ObLogReplayWorker::submit(int64_t& log_id, const char* buf, int64_t len, int64_t& pos, const ReplayType replay_type)
    {
      int err = OB_SUCCESS;
      if (!is_inited() || !replay_cursor_.is_valid())
      {
        err = OB_NOT_INIT;
      }
      else if (NULL == buf || len <= 0 || pos >= len)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else if (0 == next_submit_log_id_
               && OB_SUCCESS != (err = get_first_log_id((int64_t&)next_submit_log_id_, buf, len, pos)))
      {
        TBSYS_LOG(ERROR, "get_first_log_id()=>%d", err);
      }
      else if (0 == next_commit_log_id_
               && (next_commit_log_id_ = next_submit_log_id_) < 0)
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "first_log_id[%ld] < 0", next_submit_log_id_);
      }
      else if (OB_SUCCESS != (err = submit(worker_state_[next_submit_log_id_ % n_worker_].task_,
                                           log_id, buf, len, pos, replay_type))
               && OB_EAGAIN != err)
      {
        TBSYS_LOG(ERROR, "submit(next_submit_id=%ld)=>%d", next_submit_log_id_, err);
      }
      return err;
    }

    int ObLogReplayWorker::submit_batch(int64_t& log_id, const char* buf, int64_t len, int64_t& pos, const ReplayType replay_type)
    {
      int err = OB_SUCCESS;
      while(OB_SUCCESS == err && pos < len)
      {
        if (OB_SUCCESS != (err = submit(log_id, buf, len, pos, replay_type))
            && OB_EAGAIN != err)
        {
          TBSYS_LOG(ERROR, "submit(log_id=%ld)=>%d", log_id, err);
        }
        else if (OB_EAGAIN == err)
        {
          err = OB_SUCCESS;
        }
      }
      return err;
    }

    int replay_batch_log(ObLogReplayWorker& worker, const char* buf, int64_t len)
    {
      int err = OB_SUCCESS;
      int64_t pos = 0;
      int64_t log_id = 0;
      if (NULL == buf || 0 > len)
      {
        err = OB_INVALID_ARGUMENT;
      }
      while(OB_SUCCESS == err && pos < len)
      {
        if (OB_SUCCESS != (err = worker.submit(log_id, buf, len, pos, RT_APPLY))
            && OB_EAGAIN != err)
        {
          TBSYS_LOG(ERROR, "worker.submit()=>%d", err);
        }
        else if (OB_EAGAIN == err)
        {
          err = OB_SUCCESS;
        }
      }
      if (OB_SUCCESS != err || 0 == len)
      {}
      else if (OB_SUCCESS != (err = worker.wait_task(log_id)))
      {
        TBSYS_LOG(ERROR, "worker.wait()=>%d", err);
      }
      return err;
    }
  }; // end namespace updateserver
}; // end namespace oceanbase
