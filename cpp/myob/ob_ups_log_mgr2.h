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

#ifndef OCEANBASE_UPDATESERVER_OB_UPS_LOG_MGR_H_
#define OCEANBASE_UPDATESERVER_OB_UPS_LOG_MGR_H_

#include "common/ob_define.h"
#include "common/ob_log_cursor.h"
#include "common/ob_log_writer2.h"
#include "common/ob_role_mgr.h"
#include "ob_ups_table_mgr.h"
#include "ob_log_generator.h"
#include "ob_pos_log_reader.h"
#include "ob_ups_log_fetcher.h"
#include "ob_fetched_log.h"

namespace oceanbase
{
  namespace common
  {
    class ObFileClient;
  };
  namespace updateserver
  {
// 首先定义处理log_task需要用到的各种方法
    class IObLogApplier
    {
      public:
        IObLogApplier(){}      
        virtual ~IObLogApplier(){}
        virtual int apply_log(const common::LogCommand cmd, const char* log_data, const int64_t data_len) = 0;
    };
    class IObLogHandler: public IObLogApplier
    {
      public:
        IObLogHandler(){}
        virtual ~IObLogHandler(){}
        int check_state() const;
        virtual int response_client(const int err, const tbnet::Connection* conn, const uint32_t channel_id) = 0;
        virtual int response_master(const int err, const tbnet::Connection* conn, const uint32_t channel_id) = 0;
        virtual int send_log(const char* log_data, const int64_t data_len) = 0;
        virtual int apply_log(const common::LogCommand cmd, const char* log_data, const int64_t data_len) = 0;
    };

    class IObScpHandler
    {
      public:
        IObScpHandler(){}
        virtual ~IObScpHandler(){}
        virtual int get_file(const ObServer& master, const char* remote_dir, const char* remote_file,
                             const char* local_dir, const char* local_file, const int64_t timeout) = 0;
    };
    class IObFetchLogHandler: public IObScpHandler
    {
      public:
        IObFetchLogHandler(){}        
        virtual ~IObFetchLogHandler(){}
        virtual int get_file(const ObServer& master, const char* remote_dir, const char* remote_file,
                             const char* local_dir, const char* local_file, const int64_t timeout) = 0;
        virtual int fetch_log(const ObServer& master, const ObLogCursor& start_cursor, ObLogCursor& new_cursor,
                              char*& log_data, int64_t& data_len, const int64_t timeout) = 0;
    };

    class IResultResponsder
    {
      public:
        IResultResponsder(){}
        virtual ~IResultResponsder(){}
        virtual int response_result(const int32_t ret_code, const int32_t cmd_type, const tbnet::Connection* conn, const uint32_t channel_id) =  0;
    };

    class ObUpsLogHandler: public IObLogHandler
    {
      public:
        ObUpsLogHandler();
        virtual ~ObUpsLogHandler();
        int init(IResultResponsder* result_responsder, ObUpsTableMgr* table_mgr, common::ObSlaveMgr *slave_mgr);
        int check_state() const;
        int response_client(const int err, const tbnet::Connection* conn, const uint32_t channel_id);
        int response_master(const int err, const tbnet::Connection* conn, const uint32_t channel_id);
        int send_log(const char* log_data, const int64_t data_len);
        int apply_log(const common::LogCommand cmd, const char* log_data, const int64_t data_len);
      private:
        bool is_inited() const;
        IResultResponsder* result_responsder_;        
        ObUpsTableMgr* table_mgr_;
        common::ObSlaveMgr* slave_mgr_;        
        ObUpsMutator mutator_;
        CommonSchemaManagerWrapper schema_;
    };

    class ObUpsFetchLogHandler: public IObFetchLogHandler
    {
      public:
        ObUpsFetchLogHandler();        
        ~ObUpsFetchLogHandler();
        bool is_inited() const;        
        int init(ObFileClient* file_client, ObUpsRpcStub* rpc_stub);
        virtual int get_file(const ObServer& master, const char* remote_dir, const char* remote_file,
                             const char* local_dir, const char* local_file, const int64_t timeout);
        virtual int fetch_log(const ObServer& master, const ObLogCursor& start_cursor, ObLogCursor& new_cursor,
                              char*& log_data, int64_t& data_len, const int64_t timeout);
      private:
        ObFileClient* file_client_;
        ObUpsRpcStub* rpc_stub_;
    };
    
    class ObDummyLogHandler: public IObLogHandler
    {
      public:
        ObDummyLogHandler() {}
        virtual ~ObDummyLogHandler(){}
        virtual int response_client(const int err, const tbnet::Connection* conn, const uint32_t channel_id) {
          UNUSED(err);
          UNUSED(conn);
          UNUSED(channel_id);
          return OB_SUCCESS;
        }
        virtual int response_master(const int err, const tbnet::Connection* conn, const uint32_t channel_id) {
          UNUSED(err);
          UNUSED(conn);
          UNUSED(channel_id);
          return OB_SUCCESS;
        }
        virtual int send_log(const char* log_data, const int64_t data_len) {
          UNUSED(log_data);
          UNUSED(data_len);
          return OB_SUCCESS;
        }
        virtual int apply_log(const common::LogCommand cmd, const char* log_data, const int64_t data_len) {
          UNUSED(cmd);
          UNUSED(log_data);
          UNUSED(data_len);
          return OB_SUCCESS;
        }
    };

// 主备同步日志的框架
    class ObUpsLogMgr;
    typedef int (ObUpsLogMgr::*log_handler)(ObBatchLogEntryTask* log_task);
    class ObUpsLogMgr: public tbsys::CDefaultRunnable
    {
      friend class ObUpsLogFetcher;
      friend class ReplayRunnable;
      friend class ObMockUps;
      friend class ObUpsLogMgrTest;
      public:
        static const char* UPS_LOG_REPLAY_POINT_FILE;
        static const int UINT64_MAX_LEN = 30;
        static const int LOG_BUF_SIZE = 1<<20;
        static const int RETRY_WAIT_TIME = 10*1000;
      public:
        ObUpsLogMgr();
        virtual ~ObUpsLogMgr();

        // log_writer_, log_generator_, log_queue_以及相关线程对ObUpdateServer的其余部分完全不可见
        // table_mgr, slave_mgr, role_mgr, obi_role需要与ObUpdateServer的其余部分共享
        // result_responsder需要用来发送响应包，由ObUpdateServer实现此接口
        int init(const char* log_dir, const int64_t log_file_max_size, const int64_t log_queue_buf_size,
                 IObLogHandler* log_handler, IObFetchLogHandler* fetch_handler, common::ObRoleMgr *role_mgr, common::ObiRole* obi_role,
                 int64_t log_sync_type);
        int dump_for_debug() const;
  // 调用reset()之前，保证停掉写线程
        int reset();
        int start_service();
        void run(tbsys::CThread *thread, void *arg);

        // 对主：从log_file_id开始重放，直到log_reader返回OB_READ_NOTHING后返回，并开始启用写服务
        // 对从: 分两个阶段
        //       1. 重放本地日志, 直到log_reader返回OB_READ_NOTHING
        //       2. 向主获取剩余日志，直到追上主push过来的新日志
  // init之后，get_local_lateset_cursor()可以调用        
        int get_local_latest_cursor(const int64_t log_id_from_sst, ObLogCursor& return_cursor) const;
  // init之后，        
        int prepare_for_master(const int64_t start_replay_log_file_id);
        int prepare_for_slave(const ObServer master, SSTableMgr& sstable_mgr,
                        const ObUpsFetchParam& fetch_param, const int64_t timeout, const int64_t retry_wait_time);
  // 主响应从注册消息时有用
        int replay_this_range(const int64_t min_log_id, const int64_t max_log_id) const;
        int get_log_range_for_fetch(const int64_t max_file_id_from_sst,
                                    int64_t& min_file_id, int64_t& max_file_id) const;
        int apply_batch(const char* log_data, int64_t data_len);
        int start_fetch(const volatile bool& stop, const ObServer& server, const common::ObLogCursor& start_cursor, common::ObLogCursor& end_cursor, const int64_t timeout);
        // 供主UPS调用r，将序列化好后的日志内容放入内部缓冲区, 提交，切日志
        int fill_log(const common::LogCommand cmd, const char* log_data, const int64_t data_len,
                      tbnet::Connection* conn, uint32_t channel_id);
        int submit_log(int64_t timeout=0);
        int switch_log(uint64_t& new_log_file_id);

        // 供备UPS调用，将从主收到的一批日志放入到内部缓冲区
        int push_batch_log(const char* log_data, int64_t data_len,
                           tbnet::Connection* conn, uint32_t channel_id, int64_t timeout);

        // 等待缓冲区的所有日志都处理完毕
        int flush_log();

        int get_log(const common::ObLogCursor& log_cursor, ObFetchedLog& fetched_log, const volatile bool& stoped);
        int get_cursor(common::ObLogCursor& log_cursor) const;
        int enable(int64_t log_id);

        int64_t get_cur_log_seq() const;
        int64_t get_last_net_elapse() const;
        int64_t get_last_disk_elapse() const;
        // helper
        int write_and_flush_log2(const common::LogCommand cmd, const char* log_data, const int64_t data_len,
                      tbnet::Connection* conn, uint32_t channel_id);
        int get_replay_point(int64_t& replay_point) const;
        int write_replay_point(int64_t replay_point) const;
        void disable_role_state_check();
        void enable_role_state_check();
        
        bool debug_;
      protected:        
        void destroy_threads();
        int replay_log(const ObLogCursor& start_cursor, const ObLogCursor& end_cursor, ObLogCursor& return_cursor) const;
        int replay_for_master(const int64_t start_replay_log_file_id);
        int fetch_for_slave(const ObServer master, SSTableMgr& sstable_mgr,
                            const ObUpsFetchParam& fetch_param, const int64_t timeout, const int64_t retry_wait_time);
        int fetch_log(const ObServer master, const common::ObLogCursor& cursor, common::ObLogCursor& end_cursor, const int64_t timeout);
      private:        
        bool is_slave_master() const; 
        int get_log_queue_reader_count() const ;
        int get_work_thread_count() const;
      protected:
        int check_state() const;
        bool is_inited() const;
      protected:
        // 完成写盘，同步备，响应客户端的所有操作，
        // 仅供在彻底改造成异步方案之前使用
        int handle_log_();
        int handle_log_as_master_(ObBatchLogEntryTask* tasks);
        int handle_log_as_slave_(ObBatchLogEntryTask* tasks);

      protected: // 线程处理函数
        int handle_write(ObBatchLogEntryTask* log_task);
        int handle_response_client(ObBatchLogEntryTask* log_task);
        int handle_response_master(ObBatchLogEntryTask* log_task);
        int handle_send_log(ObBatchLogEntryTask* log_task);
        int handle_apply(ObBatchLogEntryTask* log_task);
        int handle_unexpected(ObBatchLogEntryTask* log_task);
      protected: // 调用的外部接口
        int response_client_(const int err, const tbnet::Connection* conn, const uint32_t channel_id);
        int response_master_(const int err, const tbnet::Connection* conn, const uint32_t channel_id);
        int send_log_(const char* log_data, const int64_t data_len);
        int write_log_(const char* log_data, const int64_t data_len);
        int apply_log_(const common::LogCommand cmd, const char* log_data, const int64_t data_len);
      public:        
        class LogReplayRunnable : public tbsys::CDefaultRunnable
        {
          public:
            LogReplayRunnable();
            virtual ~LogReplayRunnable();
            virtual int init(ObUpsLogMgr* log_mgr, const int64_t min_log_file_id, const int64_t max_log_file_id);
            virtual void start();
            virtual void run(tbsys::CThread* thread, void* arg);
            int wait_replay_finished(bool& stop, int64_t retry_wait_time_us);
          protected:
            ObUpsLogMgr* log_mgr_;            
            bool finished_;            
            int return_code_;
            int64_t min_log_file_id_;            
            int64_t max_log_file_id_;            
        };
        class LogFetchRunnable : public tbsys::CDefaultRunnable
        {
          public:
            LogFetchRunnable();
            virtual ~LogFetchRunnable();
            virtual int init(ObUpsLogMgr* log_mgr, const ObServer& master,  const ObLogCursor& start_cursor, const int64_t timeout);
            void destroy_threads();
            int reset();            
            virtual void run(tbsys::CThread* thread, void* arg);
          protected:
            ObUpsLogMgr* log_mgr_;            
            ObServer master_;
            ObLogCursor start_cursor_;
            int64_t timeout_;            
        };
      protected:
  // 配置及状态
        bool enable_parallel;
        bool service_started_;
        int last_error_;
        bool role_state_check_;
        int64_t log_sync_type_;

        char* log_dir_;
        int64_t log_queue_buf_size_;        
        ObLogCursor log_cursor_;

  // 私有工作类
        ObLogGenerator log_generator_;
        ObPosLogReader pos_log_reader_;
        ObUpsLogMgr::LogFetchRunnable log_fetcher_;
        ObLogWriterV2 log_writer_;
  //日志缓冲区
        ObLogQueue log_queue_;
        ObLogIterator* iters_;
  // 共享实例
        IObLogHandler* log_handler_;        
        IObFetchLogHandler* fetch_handler_;        
        ObRoleMgr* role_mgr_;
        ObiRole* obi_role_;
    };
  } // end namespace updateserver
} // end namespace oceanbase

#endif // OCEANBASE_UPDATESERVER_OB_UPS_LOG_MGR_H_
