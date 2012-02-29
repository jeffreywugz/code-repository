/**
 * (C) 2007-2010 Alibaba Group Holding Limited.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * version 2 as published by the Free Software Foundation.
 *
 * Version: $Id$
 *
 * ob_log_queue.h
 *
 * Authors:
 *   yuanqi.xhf <yuanqi.xhf@taobao.com>
 *
 */
#ifndef __OCEANBASE_UPDATESERVER_OB_LOG_QUEUE_H__
#define __OCEANBASE_UPDATESERVER_OB_LOG_QUEUE_H__

#include "ob_data_entry_queue.h"
#include "tbsys.h"
#include "tbnet.h"
#include "common/ob_log_entry.h"
#include "common/thread_buffer.h"

using namespace oceanbase::common;
namespace oceanbase
{
  namespace updateserver
  {
    struct ObLogEntryTask
    {
      int64_t log_id_;
      int packet_code;
      int err_;
      tbnet::Connection* conn_;
      uint32_t channel_id_;
      ObLogEntryTask(): log_id_(-1), packet_code(0), err_(OB_SUCCESS), conn_(NULL), channel_id_(0)
      {}
    };

    // 现在的实现中log_tasks_与log_data_直接跟在ObBatchLogEntryTask的后面
    // 相邻的ObBatchLogEntryTask的log_tasks_与log_data_在内存地址上并不相邻
    // 合并多个ObBatchLogEntryTask时需要拷贝log_tasks_与log_data_
    // 使用中的ObBatchLogEntryTask如下所示
    //          |-----------------|
    //          |                 v
    //  --- ----------- ---------- ----------------------- ----------------------
    // |...|log_tasks_ | log_data_| log_tasks_buffer .... | log_data-buffer .... |
    //  --- ----------- ---------- ----------------------- ----------------------
    //                      |                             ^
    //                      |-----------------------------|
    struct ObBatchLogEntryTask
    {
      bool for_master_;
      int64_t min_id_;
      int64_t max_id_;
      int64_t n_log_tasks_;
      int64_t log_data_len_;
      ObLogEntryTask* log_tasks_;
      char* log_data_;
      ObBatchLogEntryTask(): for_master_(true), min_id_(-1), max_id_(-1), n_log_tasks_(0), log_data_len_(0), log_tasks_(NULL), log_data_(NULL)
      {}

      bool is_valid() const
      {
        return min_id_ != -1 && max_id_ != -1 && n_log_tasks_ != 0 && log_data_len_ != 0
          && log_tasks_ != NULL && log_data_ != NULL;
      }

      int check_data_integrity() const;

      int64_t get_full_data_len() const
      {
        return sizeof(*this) + sizeof(*log_tasks_) * n_log_tasks_ + log_data_len_;
      }
      void reset()
      {
        min_id_ = -1;
        max_id_ = -1;
        n_log_tasks_ = 0;
        log_data_len_ = -1;
        log_tasks_ = NULL;
        log_data_ = NULL;
      }

      char* to_str() // for debug
      {
        static char str_buf[1024];
        snprintf(str_buf, sizeof(str_buf), "ObBatchLogEntryTask{min_id=%ld, max_id=%ld, n_log_tasks=%ld, log_tasks=%p, log_data=%p}",
                 min_id_, max_id_, n_log_tasks_, log_tasks_, log_data_);
        str_buf[sizeof(str_buf)-1] = 0;
        return str_buf;
      }

      int serialize(char* buf, int64_t len, int64_t& pos) const;
      int deserialize(char* buf, int64_t len, int64_t& pos);

    };

    // ObLogIterator.get(), ObLogBuffer.push()的timeout参数单位为微秒, 含义如下
    //  + timeout > 0: 等待timeout时间
    //  + timout == 0: 不阻塞
    //  + timeout < 0: 无限期等待

    class ObLogIterator
    {
      public:
        ObLogIterator();
        void init(ObDataEntryIterator* iter);
        // get(log_task)和commit(log_task)的含义
        //  get()获取下一条日志, 返回的log_task直接引用内部缓冲区
        //  commit()表示当前日志已经被读者使用完毕，有两个含义:
        //   + log_task所占内存可以被回收
        //   + 下次调用get()将得到下一条log_task,
        //     换言之调用commit(log_task)之前, 可以多次调用get(log_task), 获得的始终是同一条日志

        // err:
        //  + OB_SUCCESS:
        //  + OB_NEED_RETRY: 下一条日志还未产生， 需要重试
        //  + others: 其他错误，不应该重试
        //int get(ObLogEntryTask*& log_task, int64_t timeout);

        // 获得多个ObLogEntryTask
        // get_batch()保证获得的是边界对齐的多条日志
        int get(ObBatchLogEntryTask*& tasks, int64_t timeout);

        // 要保证按顺序提交
        // err:
        //   + OB_SUCCESS:
        //   + OB_INVALID_ARGUMENT: log_task为NULL, 或者与上次取出的log_task不匹配
        //   + others:
        int commit(ObBatchLogEntryTask* tasks);
      private:
        ObDataEntryIterator* data_entry_iter_;
    };

    class ObLogQueue
    {
      public:
        // 支持多读者，读者使用iterator访问，读者数目在初始化时就要指定
  // reset之后可以在此调用
        int init(int64_t size, int n_reader, ObLogIterator*& iters, int64_t retry_wait_times);

        int reset();
        //只支持单线程调用push，
        //拷贝数据到内部缓冲区, log_task可以在函数返回后重用
        // err:
        //  + OB_SUCCESS:
        //  + OB_NEED_RETRY: 缓冲区已满， 需要重试
        //  + others: 其他错误，不应该重试
        int push(const ObBatchLogEntryTask* log_tasks, int64_t timeout);

        // 启用iter前，通过iter无法获得log,
        // err:
        //  + OB_SUCCESS:
        //  + OB_NEED_RETRY: max_log_id对应的日志已经被更新的日志覆盖
        //  + others:
        // 备fetch日志线程通过这个函数判断是否已追上了主机push过来的新日志
        int enable(int64_t max_log_id);

        // 禁止buf接受新日志，并等待buf中的日志都被处理完之后返回, 在备断线之后重新注册时有用
        int flush(int64_t& max_log_id);
        // 临时方法，以后会删除
        int set_max_log_id_can_hold(int64_t max_log_id);
        int dump_for_debug() const;
      private:
        // 读者通过Iterator访问ObLogBuffer,
        // Iterator必须要在调用enable(max_id_processed)之后才可以访问。并且只能调用一次.
        ObDataEntryQueue data_entry_queue_;
        ObLogIterator iters_[OB_MAX_N_ITERATORS];
        int n_iters_;
        common::ThreadSpecificBuffer thread_buffer_;
    };
  } // end namespace updateserver
} // end namespace oceanbase
#endif // __OCEANBASE_UPDATESERVER_OB_LOG_QUEUE_H__
