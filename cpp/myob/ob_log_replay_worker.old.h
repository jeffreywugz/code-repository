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
#ifndef __OB_UPDATESERVER_OB_LOG_REPLAY_WORKER_H__
#define __OB_UPDATESERVER_OB_LOG_REPLAY_WORKER_H__
#include "common/ob_log_entry.h"
#include "ob_ups_table_mgr.h"
#include "ob_async_log_applier.h"

namespace oceanbase
{
  namespace updateserver
  {
    class ObLogReplayWorker: public tbsys::CDefaultRunnable
    {
      public:
        const static int64_t MAX_N_WORKER = 256;
        const static int64_t CHECK_SUM_BUF_SIZE = 1<<16;
        const static int64_t MAX_LOG_BUF_SIZE = 1<<22;
        struct WorkerState
        {
          ObLogTask task_;
        };
      public:
        ObLogReplayWorker();
        virtual ~ObLogReplayWorker();
        int init(TransMgr* trans_mgr_, IAsyncLogApplier* log_applier, const int64_t n_thread);
        virtual void run(tbsys::CThread* thread, void* arg);
      public:
        int64_t get_replay_log_id() const;
        int get_replay_cursor(ObLogCursor& cursor) const;
        int update_replay_cursor(const ObLogCursor& cursor);
        int start_log(const ObLogCursor& log_cursor);
        bool is_task_finished(const int64_t log_id) const;
        int wait_task(int64_t id) const;
        int alloc(char*& buf, const int64_t len);
        int release(char* buf, const int64_t len);
        int submit(int64_t& task_id, const char* buf, int64_t len, int64_t& pos, const ReplayType replay_type);
        int submit_batch(int64_t& task_id, const char* buf, int64_t len, int64_t& pos, const ReplayType replay_type);
      protected:
        bool is_inited() const;
        int submit(ObLogTask& task, int64_t& log_id, const char* buf, int64_t len, int64_t& pos, const ReplayType replay_type);
        int do_work(int64_t thread_id);
        int replay(ObLogTask& task);
      private:
        DISALLOW_COPY_AND_ASSIGN(ObLogReplayWorker);
      private:
        int64_t n_worker_;
        IAsyncLogApplier* log_applier_;
        WorkerState worker_state_[MAX_N_WORKER];        
        char log_buf_[MAX_LOG_BUF_SIZE];
        bool is_log_buf_free_;
        ObLogCursor replay_cursor_;
        TransMgr* trans_mgr_;
        volatile int64_t next_submit_log_id_;
        volatile int64_t next_commit_log_id_;
        volatile int64_t last_barrier_log_id_;
    };
    int replay_batch_log(ObLogReplayWorker& worker, const char* buf, const int64_t len);
  }; // end namespace updateserver
}; // end namespace oceanbase

#endif /* __OB_UPDATESERVER_OB_LOG_REPLAY_WORKER_H__ */
