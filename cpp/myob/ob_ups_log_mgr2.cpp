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


#include "common/utility.h"
#include "ob_ups_log_mgr2.h"
#include "ob_ups_log_mgr_utils.h"

using namespace oceanbase::common;
int parse_log_buffer(const char* log_data, int64_t data_len, const ObLogCursor& start_cursor, ObLogCursor& end_cursor);
namespace oceanbase
{
  namespace updateserver
  {

    ObUpsLogHandler::ObUpsLogHandler(): result_responsder_(NULL), table_mgr_(NULL), slave_mgr_(NULL)
    {}
    ObUpsLogHandler::~ObUpsLogHandler(){}

    bool ObUpsLogHandler::is_inited() const
    {
      return NULL != result_responsder_ && NULL != table_mgr_ && NULL != slave_mgr_;
    }

    int ObUpsLogHandler::check_state() const
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "is_inited()=>false");
      }
      return err;
    }

    int ObUpsLogHandler::init(IResultResponsder* result_responsder, ObUpsTableMgr* table_mgr, common::ObSlaveMgr *slave_mgr)
    {
      int err = OB_SUCCESS;
      if (is_inited())
      {
        err = OB_INIT_TWICE;
        TBSYS_LOG(ERROR, "is_init()=>true");
      }
      else
      {
        result_responsder_ = result_responsder;
        table_mgr_ = table_mgr;
        slave_mgr_ = slave_mgr;
      }
      return err;
    }

    int ObUpsLogHandler::response_client(const int errcode, const tbnet::Connection* conn, const uint32_t channel_id)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (OB_SUCCESS != (err = (NULL != conn? result_responsder_->response_result(errcode, OB_WRITE_RES, conn, channel_id): OB_SUCCESS)))
      {
        TBSYS_LOG(ERROR, "result_responsder->response_result(err=%ld, OB_WRITE_RES, conn=%p, channel=%d)=>%d",
                  errcode, conn, channel_id, err);
      }
      return err;
    }

    int ObUpsLogHandler::response_master(const int errcode, const tbnet::Connection* conn, const uint32_t channel_id)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (OB_SUCCESS != (err = (NULL != conn? result_responsder_->response_result(errcode, OB_SEND_LOG_RES, conn, channel_id): OB_SUCCESS)))
      {
        TBSYS_LOG(ERROR, "result_responsder->response_result(err=%ld, OB_SEND_LOG_RES, conn=%p, channel=%d)=>%d",
                  errcode, conn, channel_id, err);
      }
      return err;
    }

    int ObUpsLogHandler::send_log(const char* log_data, const int64_t data_len)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      if (OB_SUCCESS != (err = ObLogGenerator::check_log_buffer(log_data, data_len)))
      {
        TBSYS_LOG(ERROR, "check_log_buffer(log_data=%p, data_len=%ld)", log_data, data_len);
      }
      else if (OB_SUCCESS != (err = slave_mgr_->send_data(log_data, data_len)))
      {
        if (OB_PARTIAL_FAILED != err)
        {
          TBSYS_LOG(ERROR, "slave_mgr_->send_data(log_data=%p, data_len=%ld)=>%d", log_data, data_len, err);
        }
        else
        {
          TBSYS_LOG(WARN, "slave_mgr_->send_data(log_data=%p, data_len=%ld)=>%d", log_data, data_len, err);
        }
      }
      return err;
    }

    int ObUpsLogHandler::apply_log(const common::LogCommand cmd, const char* log_data, const int64_t data_len)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (OB_SUCCESS != (err = replay_single_log_func(mutator_, schema_, table_mgr_, cmd, log_data, data_len)))
      {
        TBSYS_LOG(ERROR, "table_mgr->replay_log(cmd=%d, log_data=%p, data_len=%ld)=>%d",
                  cmd, log_data, data_len, err);
      }
      return err;
    }

    ObUpsFetchLogHandler:: ObUpsFetchLogHandler(): file_client_(NULL), rpc_stub_(NULL)
    {}

    ObUpsFetchLogHandler:: ~ObUpsFetchLogHandler()
    {}

    bool ObUpsFetchLogHandler:: is_inited() const
    {
      return NULL != file_client_ && NULL != rpc_stub_;
    }

    int ObUpsFetchLogHandler:: init(ObFileClient* file_client, ObUpsRpcStub* rpc_stub)
    {
      int err = OB_SUCCESS;
      if (is_inited())
      {
        err = OB_INIT_TWICE;
        TBSYS_LOG(ERROR, "init twice");
      }
      else if (NULL == file_client || NULL == rpc_stub)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "file_client=%p,rpc_stub=%p", file_client, rpc_stub);
      }
      else
      {
        file_client_ = file_client;
        rpc_stub_ = rpc_stub;
      }
      return err;
    }

    int ObUpsFetchLogHandler:: get_file(const ObServer& master, const char* remote_dir, const char* remote_file,
                                        const char* local_dir, const char* local_file, const int64_t timeout)
    {
      int err = OB_SUCCESS;
      bool use_scp = true;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "not init");
      }
      else if (OB_SUCCESS != (err = cp_remote_file_func(use_scp? NULL: file_client_, master, remote_dir, remote_file, local_dir, local_file, timeout)))
      {
        TBSYS_LOG(ERROR, "cp_remote_file()=>%d", err);
      }
      return err;
    }

    int ObUpsFetchLogHandler:: fetch_log(const ObServer& master, const ObLogCursor& start_cursor, ObLogCursor& end_cursor,
                                         char*& log_data, int64_t& data_len, const int64_t timeout)
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "not init");
      }
      UNUSED(master);
      UNUSED(start_cursor);
      UNUSED(end_cursor);
      UNUSED(log_data);
      UNUSED(data_len);
      UNUSED(timeout);
      // else if (OB_SUCCESS != (err = rpc_stub_->fetch_log(master, start_cursor, end_cursor, log_data, data_len, timeout)))
      // {
      //   if (OB_NEED_RETRY != err)
      //   {
      //     TBSYS_LOG(ERROR, "rpc_stub->fetch_log()=>%d", err);
      //   }
      // }
      return err;
    }

    const char* ObUpsLogMgr::UPS_LOG_REPLAY_POINT_FILE = "log_replay_point";
    ObUpsLogMgr::ObUpsLogMgr(): debug_(false), enable_parallel(false), service_started_(false), last_error_(OB_SUCCESS),
                                role_state_check_(false), log_sync_type_(OB_LOG_SYNC),
                                log_dir_(NULL), log_queue_buf_size_(-1), log_cursor_(),
                                log_handler_(NULL), fetch_handler_(NULL), role_mgr_(NULL), obi_role_(NULL)
    {
    }

    ObUpsLogMgr::~ObUpsLogMgr()
    {
      TBSYS_LOG(INFO, "log_dir=%p", log_dir_);
      if (NULL != log_dir_)
      {
        free(log_dir_);
        log_dir_ = NULL;
      }
    }

    int ObUpsLogMgr::init(const char* log_dir, const int64_t log_file_max_size,
                          const int64_t log_queue_buf_size,
                          IObLogHandler* log_handler, IObFetchLogHandler* fetch_handler, common::ObRoleMgr *role_mgr, common::ObiRole* obi_role,
                          int64_t log_sync_type)
    {
      int err = OB_SUCCESS;
      int64_t n_cached_log_reader = 1<<4;
      int64_t session_duration_us = 10 * 1000 * 1000;
      if (is_inited())
      {
        err = OB_INIT_TWICE;
        TBSYS_LOG(ERROR, "init twice.");
      }
      else if (NULL == log_dir || NULL == log_handler || NULL == fetch_handler || NULL == role_mgr || NULL == obi_role
               || 0 >= log_file_max_size || 0 >= log_queue_buf_size)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "init(log_dir=%p, log_file_max_size=%ld, log_queue_buf_size=%ld, log_handler=%p, role_mgr=%p, obi_role=%p",
                  log_dir, log_file_max_size, log_queue_buf_size, role_mgr, obi_role);
      }
      else if (OB_SUCCESS != (err = log_writer_.init(log_dir, log_sync_type)))
      {
        TBSYS_LOG(ERROR, "log_writer_.init(log_dir=%s)=>%d", log_dir, err);
      }
      else if (OB_SUCCESS != (err = log_generator_.init(LOG_BUF_SIZE, log_file_max_size)))
      {
        TBSYS_LOG(ERROR, "log_generator_.init(log_buf_size=%ld, log_file_max_size=%ld)=>%d",
                  LOG_BUF_SIZE, log_file_max_size, err);
      }
      else if (OB_SUCCESS != (err = pos_log_reader_.init(log_dir, n_cached_log_reader, session_duration_us)))
      {
        TBSYS_LOG(ERROR, "pos_log_reader.init(log_dir=%s)=>%d", log_dir, err);
      }
      else if (NULL == (log_dir_ = strndup(log_dir, OB_MAX_FILE_NAME_LENGTH)))
      {
        TBSYS_LOG(ERROR, "strdup(%s)=>NULL", log_dir);
      }
      else
      {
        TBSYS_LOG(INFO, "log_dir=%p", log_dir_);
        last_error_ = OB_SUCCESS;
        role_state_check_ = true;

        log_queue_buf_size_ = log_queue_buf_size;
        log_sync_type_ = log_sync_type;

        log_handler_ = log_handler;
        fetch_handler_ = fetch_handler;
        role_mgr_ = role_mgr;
        obi_role_ = obi_role;
      }
      return err;
    }

    bool ObUpsLogMgr::is_slave_master() const
    {
      return (ObRoleMgr::MASTER == role_mgr_->get_role() && ObiRole::SLAVE == obi_role_->get_role());
    }

    int ObUpsLogMgr::get_log_queue_reader_count() const
    {
      return enable_parallel? (is_slave_master()? 4: 3): 1;
    }

    int ObUpsLogMgr::get_work_thread_count() const
    {
      return enable_parallel? (is_slave_master()? 4: 3): 0;
    }

    int ObUpsLogMgr::get_replay_point(int64_t& replay_point) const
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "is_inited()=>false");
      }
      else if (OB_SUCCESS != (err = get_replay_point_func(log_dir_, replay_point)))
      {
        TBSYS_LOG(ERROR, "get_replay_point_func(log_dir=%s)=>%d", log_dir_, err);
      }
      return err;
    }


    bool ObUpsLogMgr::is_inited() const
    {
      return NULL != log_handler_ && NULL != role_mgr_ && NULL != obi_role_;
    }

    int ObUpsLogMgr::check_state() const
    {
      int err = OB_SUCCESS;
      if (!is_inited() || !log_cursor_.is_valid())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "check_state(is_inited()=%s, log_cursor=%s)=>%d", STR_BOOL(is_inited()), log_cursor_.to_str(), err);
      }
      else
      {
        err = last_error_;
      }
      return err;
    }

    int ObUpsLogMgr::dump_for_debug() const
    {
      int err = OB_SUCCESS;
      TBSYS_LOG(INFO, "UpsLogMgr::log_cursor=%s", log_cursor_.to_str());
      log_queue_.dump_for_debug();
      return err;
    }

    void ObUpsLogMgr::disable_role_state_check()
    {
      role_state_check_ = false;
    }

    void ObUpsLogMgr::enable_role_state_check()
    {
      role_state_check_ = true;
    }

    void ObUpsLogMgr::destroy_threads()
    {
      if (_thread) {
        delete[] _thread;
        _thread = NULL;
      }
      _stop = false;
    }

    int ObUpsLogMgr::reset()
    {
      int err = OB_SUCCESS;
      int64_t max_id = 0;
      TBSYS_LOG(INFO, "log_mgr.reset: %s", log_cursor_.to_str());
      if (!is_inited())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (OB_SUCCESS != (err = log_fetcher_.reset()))
      {
        TBSYS_LOG(ERROR, "log_fetcher.reset()=>%d", err);
      }
      else if (service_started_ && OB_SUCCESS != (err = log_queue_.flush(max_id))) // may block
      {
        TBSYS_LOG(ERROR, "log_queue.flush()=>%d", err);
      }
      else
      {
        stop();
        wait();
      }

      if (OB_SUCCESS != err)
      {}
      else if (OB_SUCCESS != (err = log_generator_.reset()))
      {
        TBSYS_LOG(ERROR, "log_writer.reset()=>%d", err);
      }
      else if (OB_SUCCESS != (err = log_writer_.reset()))
      {
        TBSYS_LOG(ERROR, "log_writer.reset()=>%d", err);
      }
      else if (service_started_ && OB_SUCCESS != (err = log_queue_.reset()))
      {
        TBSYS_LOG(ERROR, "log_queue.reset()=>%d", err);
      }
      // else if (OB_SUCCESS != (err = log_generator_.start_log(log_cursor_)))
      // {
      //   TBSYS_LOG(ERROR, "log_generator.start_log(%s)=>%d", log_cursor_.to_str(), err);
      // }
      // else if (OB_SUCCESS != (err = log_writer_.start_log(log_cursor_)))
      // {
      //   TBSYS_LOG(ERROR, "log_writer.start_log(%s)=>%d", log_cursor_.to_str(), err);
      // }
      else
      {
        last_error_ = OB_SUCCESS;
        role_state_check_ = true;
      }
      destroy_threads();
      return err;
    }

    int ObUpsLogMgr::get_cursor(ObLogCursor& log_cursor) const
    {
      int err = OB_SUCCESS; 
     if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else
      {
        log_cursor = log_cursor_;
      }
      return err;
    }

    int ObUpsLogMgr::write_replay_point(int64_t replay_point) const
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "is_inited()=>false");
      }
      else if (OB_SUCCESS != (err= write_replay_point_func(log_dir_, replay_point)))
      {
        TBSYS_LOG(ERROR, "write_replay_point_func(log_dir=%s, replay_point=%ld)=>%d", log_dir_, replay_point, err);
      }
      return err;
    }

    int ObUpsLogMgr::get_log_range_for_fetch(const int64_t max_file_id_from_sst,
                                             int64_t& min_file_id, int64_t& max_file_id) const
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "log_mgr not init.");
      }
      else if (!log_cursor_.is_valid())
      {
        err = OB_NEED_RETRY;
        TBSYS_LOG(DEBUG, "log_cursor is invalid, need retry");
      }
      else if (OB_INVALID_ID != (uint64_t)max_file_id_from_sst)
      {
        min_file_id = max_file_id_from_sst;
      }
      else if (OB_SUCCESS != (err = get_replay_point(min_file_id)))
      {
        TBSYS_LOG(ERROR, "log_mgr.get_replay_point()=>%d", err);
      }

      if (OB_SUCCESS == err)
      {
        max_file_id = log_cursor_.file_id_;
      }
      return err;
    }

    int ObUpsLogMgr::prepare_for_master(const int64_t start_replay_log_file_id)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = replay_for_master(start_replay_log_file_id)))
      {
        TBSYS_LOG(ERROR, "log_mgr_.replay_log_for_master(replay_id=%ld)=>%d", start_replay_log_file_id, err);
      }
      else if (OB_SUCCESS != (err = start_service()))
      {
        TBSYS_LOG(ERROR, "log_mgr_.start_service()=>%d", err);
      }
      else if (OB_SUCCESS != (err = enable(0))) // 可以向缓冲区中写
      {
        TBSYS_LOG(ERROR, "log_mgr_.enable(0)=>%d", err);
      }
      return err;
    }

    int ObUpsLogMgr::prepare_for_slave(const ObServer master, SSTableMgr& sstable_mgr,
                        const ObUpsFetchParam& fetch_param, const int64_t timeout, const int64_t retry_wait_time)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = fetch_for_slave(master, sstable_mgr, fetch_param, timeout, retry_wait_time)))
      {
        TBSYS_LOG(ERROR, "fetch_for_slave()=>%d", err);
      }
      return err;
    }

    int ObUpsLogMgr::get_local_latest_cursor(const int64_t log_id_from_sst, ObLogCursor& return_cursor) const
    {
      int err = OB_SUCCESS;
      int64_t replay_point = 0;
      int64_t min_log_id = 0;
      int64_t max_log_id = 0;
      ObDummyLogHandler log_handler;
      ObLogCursor start_cursor;
      ObLogCursor end_cursor;
      UNUSED(log_id_from_sst);
      if (!is_inited())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (OB_SUCCESS != (err = scan_log_dir_func(log_dir_, replay_point, min_log_id, max_log_id)))
      {
        TBSYS_LOG(ERROR, "scan_log_dir_func(log_dir=%s)=>%d", log_dir_, err);
      }
      else
      {
        start_cursor.file_id_ = max_log_id;
      }

      if (OB_SUCCESS != err)
      {}
      else if (OB_SUCCESS != (err = replay_log_func(_stop, log_dir_, start_cursor, end_cursor, return_cursor, &log_handler)))
      {
        TBSYS_LOG(ERROR, "replay_log(log_cursor=%s)=>%d", log_cursor_.to_str(), err);
      }

      return err;
    }

    int ObUpsLogMgr::replay_for_master(const int64_t log_id)
    {
      int err = OB_SUCCESS;
      ObLogCursor start_cursor;
      ObLogCursor end_cursor;
      ObLogCursor return_cursor;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (!log_cursor_.is_valid())
      {
        if (OB_INVALID_ID != (uint64_t)log_id)
        {
          start_cursor.file_id_ = log_id;
        }
        else if (OB_SUCCESS != (err = get_replay_point_func(log_dir_, start_cursor.file_id_)))
        {
          TBSYS_LOG(ERROR, "get_replay_point_func(log_dir=%s)=>%d", log_dir_, err);
        }

        if (OB_SUCCESS != err)
        {}
        else if (OB_SUCCESS != (err = replay_log_func(_stop, log_dir_, start_cursor, end_cursor, log_cursor_, log_handler_)))
        {
          TBSYS_LOG(ERROR, "replay_log(log_cursor=%s)=>%d", log_cursor_.to_str(), err);
        }
      }

      if (OB_SUCCESS != err)
      {}
      else if (OB_SUCCESS != (err = log_generator_.start_log(log_cursor_)))
      {
        TBSYS_LOG(ERROR, "log_generator.start_log(%s)=>%d", log_cursor_.to_str(), err);
      }
      else if (OB_SUCCESS != (err = log_writer_.start_log(log_cursor_)))
      {
        TBSYS_LOG(ERROR, "log_writer.start_log(%s)=>%d", log_cursor_.to_str(), err);
      }
      return err;
    }

    int ObUpsLogMgr::replay_log(const ObLogCursor& start_cursor, const ObLogCursor& end_cursor, ObLogCursor& return_cursor) const
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        TBSYS_LOG(ERROR, "is_inited()=>false");
      }
      else if(OB_SUCCESS != (err = replay_log_func(_stop, log_dir_, start_cursor, end_cursor, return_cursor, log_handler_)))
      {
        TBSYS_LOG(ERROR, "replay_log_func(log_dir=%s, start_cursor=%s)=>%d", log_dir_, start_cursor.to_str(), err);
      }
      return err;
    }

    int ObUpsLogMgr::replay_this_range(const int64_t min_file_id, const int64_t max_file_id) const
    {
      int err = OB_SUCCESS;
      ObLogCursor start_cursor;
      ObLogCursor end_cursor;
      ObLogCursor return_cursor;
      if (!is_inited())
      {
        TBSYS_LOG(ERROR, "is_inited()=>false");
      }
      else
      {
        start_cursor.file_id_ = min_file_id;
        end_cursor.file_id_ = max_file_id;
      }

      if (OB_SUCCESS != err)
      {}
      else if(OB_SUCCESS != (err = replay_log(start_cursor, end_cursor, return_cursor)))
      {
        TBSYS_LOG(ERROR, "replay_log(log_dir=%s, min_id=%ld, max_id=%ld)=>%d", log_dir_, min_file_id, max_file_id, err);
      }
      return err;
    }

    int fill_fetch_param_for_fetch(const char* log_dir, const ObUpsFetchParam& fetch_param, ObUpsFetchParam& new_fetch_param)
    {
      int err = OB_SUCCESS;
      int64_t replay_point = 0;
      int64_t min_log_id = 0;
      int64_t max_log_id = 0;
      if (NULL == log_dir)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "log_dir == NULL");
      }
      else if (OB_SUCCESS != (err = scan_log_dir_func(log_dir, replay_point, min_log_id, max_log_id)))
      {
        TBSYS_LOG(ERROR, "scan_log_dir(log_dir=%s)=>%d", log_dir);
      }
      else
      {
        new_fetch_param.clone(fetch_param);
        if (fetch_param.min_log_id_ < (uint64_t)max_log_id)
        {
          new_fetch_param.min_log_id_ = max_log_id;
        }
      }
      return err;
    }

    int ObUpsLogMgr::fetch_for_slave(const ObServer master, SSTableMgr& sstable_mgr,
                                     const ObUpsFetchParam& fetch_param, const int64_t timeout, const int64_t retry_wait_time)
    {
      int err = OB_SUCCESS;
      int replay_err = OB_SUCCESS;
      bool replay_start = false;
      ObUpsFetchParam new_fetch_param;
      char buf[1024];
      if (!is_inited() || _stop)
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "is_inited()=%s || stop=%s", STR_BOOL(is_inited()), STR_BOOL(_stop));
      }
      else
      {
        if (OB_SUCCESS != (err = fetch_param.to_str(buf, sizeof(buf))))
        {
          TBSYS_LOG(WARN, "fetch_param.to_str(len=%ld)=>%d", sizeof(buf), err);
        }
        else
        {
          TBSYS_LOG(INFO, "fetch_for_slave(fetch_param=[%s])", buf);
        }
      }

      if (OB_SUCCESS != err)
      {}
      else if (!log_cursor_.is_valid())
      {
        log_cursor_.file_id_ = fetch_param.max_log_id_;
        ObLogCursor start_cursor = log_cursor_;
        ObLogCursor end_cursor;
        ObUpsLogMgr::LogReplayRunnable replay_runnable;
        if (OB_SUCCESS != (err = replay_runnable.init(this, fetch_param.min_log_id_, fetch_param.max_log_id_)))
        {
          TBSYS_LOG(ERROR, "replay_runnable.init(log_file_range=[%ld, %ld])=>%d",
                    fetch_param.min_log_id_, fetch_param.max_log_id_, err);
        }
        else
        {
          replay_runnable.start();
          replay_start = true;
        }

        if (OB_SUCCESS != err)
        {}
        else if (OB_SUCCESS != (err = fill_fetch_param_for_fetch(log_dir_, fetch_param, new_fetch_param)))
        {
          TBSYS_LOG(ERROR, "fill_fetch_param_for_fetch(log_dir=%s)", log_dir_);
        }
        else if (OB_SUCCESS != (err = fetch_sst_and_log_file_func(_stop, fetch_handler_, master, log_dir_, sstable_mgr.get_store_mgr(), new_fetch_param, timeout)))
        {
          TBSYS_LOG(ERROR, "fetch_sst_and_log_file()=>%d", err);
        }
        else
        {
          sstable_mgr.reload_all();
        }
        
        if (replay_start && OB_SUCCESS != (replay_err = replay_runnable.wait_replay_finished(_stop, retry_wait_time)))
        {
          err = replay_err;
          TBSYS_LOG(ERROR, "replay_runnable.wait_replay_finished()=>%d", err);
        }

        if (OB_SUCCESS != err)
        {}
        else if (OB_SUCCESS != (err = replay_log(start_cursor, end_cursor, log_cursor_)))
        {
          TBSYS_LOG(ERROR, "replay_log(log_cursor=%s)=>%d", log_cursor_.to_str(), err);
        }
        else
        {
          TBSYS_LOG(INFO, "log_mgr: start_log(%s)", log_cursor_.to_str());
        }
      }

      if (OB_SUCCESS != err)
      {}
      else if (OB_SUCCESS != (err = log_generator_.start_log(log_cursor_)))
      {
        TBSYS_LOG(ERROR, "log_generator.start_log(%s)=>%d", log_cursor_.to_str(), err);
      }
      else if (OB_SUCCESS != (err = log_writer_.start_log(log_cursor_)))
      {
        TBSYS_LOG(ERROR, "log_writer.start_log(%s)=>%d", log_cursor_.to_str(), err);
      }
      else if (OB_SUCCESS != (err = start_service()))
      {
        TBSYS_LOG(ERROR, "log_mgr_.start_service()=>%d", err);
      }
      else if (OB_SUCCESS != (err = log_fetcher_.init(this, master, log_cursor_, timeout)))
      {
        TBSYS_LOG(ERROR, "log_fetcher_.init()=>%d", err);
      }
      else if (OB_SUCCESS == err)
      {
        log_fetcher_.start();
      }
      return err;
    }

    int ObUpsLogMgr:: get_log(const common::ObLogCursor& log_cursor, ObFetchedLog& fetched_log, const volatile bool& stoped)
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "ob_ups_log_mgr not init.");
      }
      // else if (OB_SUCCESS != (err = pos_log_reader_.get_log(log_cursor, fetched_log, stoped)))
      // {
      //   if (OB_READ_NOTHING == err)
      //   {
      //     TBSYS_LOG(DEBUG, "pos_log_reader.get_log(log_cursor=%s)=>%d", log_cursor.to_str(), err);
      //   }
      //   else
      //   {
      //     TBSYS_LOG(ERROR, "pos_log_reader.get_log(log_cursor=%s)=>%d", log_cursor.to_str(), err);
      //   }
      // }
      UNUSED(log_cursor);
      UNUSED(fetched_log);
      if (stoped)
      {
      }
      err = OB_NOT_SUPPORTED;
      return err;
    }

  int ObUpsLogMgr::fill_log(const LogCommand cmd, const char* log_data, const int64_t data_len,
                           tbnet::Connection* conn, uint32_t channel_id)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (OB_SUCCESS != (err = log_generator_.write_log(cmd, log_data, data_len, conn, channel_id)))
      {
        if (OB_BUF_NOT_ENOUGH != err)
        {
          TBSYS_LOG(ERROR, "log_generator_.write_log()=>%d", err);
        }
        else
        {
          TBSYS_LOG(DEBUG, "log_generator_.write_log()=>%d", err);
        }
      }
      // else if (OB_SUCCESS != (err = log_generator_.get_cursor(log_cursor_)))
      // {
      //   TBSYS_LOG(ERROR, "log_generator.get_cursor()=>%d", err);
      // }
      if (OB_SUCCESS != err && OB_BUF_NOT_ENOUGH != err)
      {
        last_error_ = err;
      }
      return err;
    }

    int ObUpsLogMgr::switch_log(uint64_t& new_log_file_id)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (OB_SUCCESS != (err = log_generator_.switch_log(new_log_file_id, NULL, 0)))
      {
        TBSYS_LOG(ERROR, "log_generator_.switch_log()=>%d", err);
      }
      else if (OB_SUCCESS != (err = submit_log()))
      {
        TBSYS_LOG(ERROR, "submit_log()=>%d", err);
      }
      if (OB_SUCCESS != err)
      {
        last_error_ = err;
      }
      return err;
    }

    int ObUpsLogMgr:: submit_log(int64_t timeout)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (OB_SUCCESS != (err = log_generator_.push_to(log_queue_, timeout)))
      {
        if (OB_NEED_RETRY != err)
        {
          TBSYS_LOG(ERROR, "log_generator.push_to(log_queue)=>%d", err);
        }
      }
      // else if (OB_SUCCESS != (err = handle_log_()))
      // {
      //   TBSYS_LOG(ERROR, "handle_log()=>%d", err);
      // }
      if (OB_READ_NOTHING == err)
      {
        err = OB_SUCCESS;
      }
      if (OB_SUCCESS != err && OB_NEED_RETRY != err)
      {
        last_error_ = err;
      }
      return err;
    }

    int ObUpsLogMgr::push_batch_log(const char* log_data, int64_t data_len,
                                 tbnet::Connection* conn, uint32_t channel_id, int64_t timeout)
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "log_mgr not init.");
      }
      else if (!service_started_)
      {
        TBSYS_LOG(WARN, "log_mgr.push_batch_log() buf not avaiable!");
      }
      else if (OB_SUCCESS != (err = log_generator_.push_batch_log(log_data, data_len, conn, channel_id, timeout)))
      {
        TBSYS_LOG(ERROR, "log_generator_.push_batch_log(log_data=%p, data_len=%ld)=>%d", log_data, data_len, err);
      }
      else if (OB_SUCCESS != (err = submit_log(timeout)))
      {
        TBSYS_LOG(ERROR, "submit_log()=>%d", err);
      }
      if (OB_SUCCESS != err && OB_NEED_RETRY != err)
      {
        last_error_ = err;
      }
      return err;
    }

    int ObUpsLogMgr:: flush_log()
    {
      int err = OB_SUCCESS;
      int64_t max_log_id = 0;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (OB_SUCCESS != (err = log_queue_.flush(max_log_id)))
      {
        TBSYS_LOG(ERROR, "flush()=>%d", err);
      }
      else if (OB_SUCCESS != (err = log_queue_.set_max_log_id_can_hold(-1)))
      {
        TBSYS_LOG(ERROR, "log_queue_.set_max_log_id_can_holde(-1)=>%d", err);
      }
      if (OB_SUCCESS != err)
      {
        last_error_ = err;
      }
      return err;
    }

    int ObUpsLogMgr:: handle_log_()
    {
      int err = OB_SUCCESS;
      ObBatchLogEntryTask* tasks = NULL;
      ObLogCursor end_cursor;
      int64_t end_pos = 0;
      int64_t timeout = 0;
      ObRoleMgr::State state = ObRoleMgr::INIT;
      //TBSYS_LOG(INFO, "handle_log()");
       if (!is_inited())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "log_mgr not init.");
      }
      else if (ObRoleMgr::ERROR == (state = role_mgr_->get_state()))
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "role_mgr.state=%s", role_mgr_->get_state_str());
      }
      // else if (ObRoleMgr::STOP == state)
      // {
      //   err = OB_NEED_RETRY;
      //   //TBSYS_LOG(INFO, "server request stop");
      // }
      else if (ObRoleMgr::INIT == state)
      {
        err = OB_NEED_RETRY;
      }
      // else if (role_state_check_ && ObRoleMgr::ACTIVE != state)
      // {
      //   err = OB_ERR_UNEXPECTED;
      //   TBSYS_LOG(ERROR, "role_mgr_.get_state() != ACTIVE");
      // }
      else if (OB_SUCCESS != (err = iters_[0].get(tasks, timeout)))
      {
        if (OB_NEED_RETRY != err)
        {
          TBSYS_LOG(ERROR, "iters[0].get(log_tasks)=>%d", err);
        }
        else
        {
          TBSYS_LOG(DEBUG, "iter[0].get()=>%d", err);
        }
      }
      else if (OB_SUCCESS != (err = tasks->check_data_integrity()))
      {
        TBSYS_LOG(ERROR, "tasks.check_data_integrity()=>%d", err);
        log_queue_.dump_for_debug();
      }
      else if (OB_SUCCESS != (err = crop_log_task(tasks, log_cursor_.log_id_)))
      { // 不要和parse_log_buffer颠倒顺序
        TBSYS_LOG(ERROR, "crop_log_task(tasks[min_id=%ld, max_id=%ld], cursor.log_id=%ld)=>%d", tasks->min_id_, tasks->max_id_, log_cursor_.log_id_, err);
      }
      else if (OB_SUCCESS != (err = parse_log_buffer(tasks->log_data_, tasks->log_data_len_, tasks->log_data_len_,
                                                     end_pos, log_cursor_, end_cursor)))
      {
        TBSYS_LOG(ERROR, "parse_log_buffer(data=%p, len=%ld, start_cursor=%s)=>%d",
                  tasks->log_data_, tasks->log_data_len_, log_cursor_.to_str(), err);
      }
      else if (ObiRole::MASTER == obi_role_->get_role() && ObRoleMgr::MASTER == role_mgr_->get_role())
      {
        if (OB_SUCCESS != (err = handle_log_as_master_(tasks)))
        {
          TBSYS_LOG(ERROR, "handle_log_as_master()=>%d", err);
        }
      }
      else
      {
        if (OB_SUCCESS != (err = handle_log_as_slave_(tasks)))
        {
          TBSYS_LOG(ERROR, "handle_log_as_slave()=>%d", err);
        }
      }

      if (OB_SUCCESS != err)
      {}
      else if (OB_SUCCESS != (err = iters_[0].commit(tasks)))
      {
        TBSYS_LOG(ERROR, "iters_[0].commit(min_id=%ld, max_id=%ld)=>%d", tasks->min_id_, tasks->max_id_, err);
      }
      else
      {
        log_cursor_ = end_cursor;
      }

      if (OB_NEED_RETRY == err)
      {
        err = OB_SUCCESS;
      }
      return err;
    }

    int ObUpsLogMgr:: handle_log_as_master_(ObBatchLogEntryTask* tasks)
    {
      int err = OB_SUCCESS;
      //TBSYS_LOG(INFO, "handle_as_master()");
      if (NULL == tasks)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "handl_log_as_master(tasks=NULL)");
      }
      else if (debug_ && OB_SUCCESS != (err = handle_apply(tasks)))
      {
         TBSYS_LOG(ERROR, "write_log()=>%d", err);
      }
      else if (OB_SUCCESS != (err = handle_write(tasks)))
      {
         TBSYS_LOG(ERROR, "write_log()=>%d", err);
      }
      else if (OB_SUCCESS != (err = handle_send_log(tasks)))
      {
        TBSYS_LOG(ERROR, "send_log()=>%d", err);
      }
      else if (OB_SUCCESS != (err = handle_response_client(tasks)))
      {
        TBSYS_LOG(ERROR, "handle_response_client()=>%d", err);
      }
      if (OB_NEED_RETRY == err)
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "handle log must be done successful.");
      }
      return err;
    }

    int ObUpsLogMgr:: handle_log_as_slave_(ObBatchLogEntryTask* tasks)
    {
      int err = OB_SUCCESS;
      //TBSYS_LOG(INFO, "handle_as_slave([%ld, %ld])", tasks->min_id_, tasks->max_id_);
      bool is_slave_master = (ObRoleMgr::MASTER == role_mgr_->get_role() && ObiRole::SLAVE == obi_role_->get_role());
      if (NULL == tasks)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "handl_log_as_slave(tasks=NULL)");
      }
      else if (OB_SUCCESS != (err = handle_write(tasks)))
      {
        TBSYS_LOG(ERROR, "write_log()=>%d", err);
      }
      else if (OB_SUCCESS != (err = handle_apply(tasks)))
      {
        TBSYS_LOG(ERROR, "handle_apply()=>%d", err);
      }
      else if (is_slave_master && OB_SUCCESS != (err = handle_send_log(tasks)))
      {
        TBSYS_LOG(ERROR, "send_log()=>%d", err);
      }
      else if (OB_SUCCESS != (err = handle_response_master(tasks)))
      {
        TBSYS_LOG(ERROR, "handle_response_master()=>%d", err);
      }
      if (OB_NEED_RETRY == err)
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "handle log must be done successful.");
      }
      return err;
    }

    int ObUpsLogMgr::start_service()
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (OB_SUCCESS != (err = log_queue_.init(log_queue_buf_size_, get_log_queue_reader_count(), iters_, RETRY_WAIT_TIME)))
      {
        TBSYS_LOG(ERROR, "log_queue.init(queue_buf_size=%ld, n_reader=%d, retry_wait_time=%ld)=>%d",
                  log_queue_buf_size_, get_log_queue_reader_count(), RETRY_WAIT_TIME, err);
      }
      else
      {
        setThreadCount(get_log_queue_reader_count());
        _stop = false;
        start();
      }
      TBSYS_LOG(INFO, "log_mgr.start_service()=>%d", err);
      return err;
    }

    void ObUpsLogMgr:: run(tbsys::CThread* thread, void* arg)
    {
      int err = OB_SUCCESS;
      ObBatchLogEntryTask* log_task = NULL;
      bool is_master = (ObRoleMgr::MASTER == role_mgr_->get_role() && ObiRole::MASTER == obi_role_->get_role());
      bool is_slave_master = (ObRoleMgr::MASTER == role_mgr_->get_role() && ObiRole::SLAVE == obi_role_->get_role());
      log_handler handler[] = {
        is_master? &ObUpsLogMgr::handle_response_client: &ObUpsLogMgr::handle_response_master,
        &ObUpsLogMgr::handle_write,
        is_master? &ObUpsLogMgr::handle_send_log: &ObUpsLogMgr::handle_apply,
        is_slave_master? &ObUpsLogMgr::handle_send_log: &ObUpsLogMgr::handle_unexpected,
      };
      if (!enable_parallel)
      {
        handler[0] = is_master? &ObUpsLogMgr::handle_log_as_master_: &ObUpsLogMgr::handle_log_as_slave_;
      }
      int64_t id = (int64_t)arg;
      UNUSED(thread);
      service_started_ = true;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }

      if (!enable_parallel)
      {
        while(!_stop && (OB_SUCCESS == err || OB_NEED_RETRY == err))
        {
          if (OB_SUCCESS != (err = handle_log_()) && OB_NEED_RETRY != err)
          {
            TBSYS_LOG(ERROR, "handle_log()=>%d", err);
          }
        }
      }
      else
      {
        while(!_stop && (OB_SUCCESS == err || OB_NEED_RETRY == err))
        {
          if (OB_SUCCESS != (err = iters_[id].get(log_task, 0)))
          {
            if (OB_NEED_RETRY != err)
            {
              TBSYS_LOG(ERROR, "iters[%ld].get(log_task)=>%d", id, err);
            }
          }
          else if (OB_SUCCESS != (err = (this->*handler[id])(log_task)))
          {
            TBSYS_LOG(ERROR, "handler[%ld](log_task)=>%d", id, err);
          }
          else if (OB_SUCCESS != (err = (iters_[id].commit(log_task))))
          {
            TBSYS_LOG(ERROR, "iters[%ld].commit(log_task[%ld, %ld])=>%d", id, log_task->min_id_, log_task->max_id_, err);
          }
        }
      }
      if (OB_SUCCESS != err)
      {
        last_error_ = err;
        if (NULL != role_mgr_)
        {
          role_mgr_->set_state(ObRoleMgr::ERROR);
        }
      }
      service_started_ = false;
    }

    int ObUpsLogMgr::enable(int64_t log_id)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (OB_SUCCESS != (err = log_queue_.enable(log_id)) && OB_NEED_RETRY != err)
      {
        TBSYS_LOG(ERROR, "log_queue.enable(log_id=%ld)=>%d", log_id, err);
      }
      return err;
    }

    int ObUpsLogMgr:: response_client_(const int err, const tbnet::Connection* conn, const uint32_t channel_id)
    {
      return log_handler_->response_client(err, conn, channel_id);
    }

    int ObUpsLogMgr:: response_master_(const int err, const tbnet::Connection* conn, const uint32_t channel_id)
    {
      return log_handler_->response_master(err, conn, channel_id);
    }

    int ObUpsLogMgr:: send_log_(const char* log_data, int64_t data_len)
    {
      return log_handler_->send_log(log_data, data_len);
    }

    int ObUpsLogMgr:: write_log_(const char* log_data, int64_t data_len)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = log_writer_.write_log(log_data, data_len)))
      {
        TBSYS_LOG(ERROR, "write_log(buf=%p, len=%ld)=>%d", log_data, data_len, err);
      }
      else if(OB_SUCCESS != (err = log_writer_.get_cursor(log_cursor_)))
      {
        TBSYS_LOG(ERROR, "ObLogWriter::get_log_cursor(%s)=>%d", log_cursor_.to_str(), err);
      }
      return err;
    }

    int ObUpsLogMgr:: apply_log_(const LogCommand cmd, const char* log_data, const int64_t data_len)
    {
      return log_handler_->apply_log(cmd, log_data, data_len);
    }

    int ObUpsLogMgr:: handle_unexpected(ObBatchLogEntryTask* log_task)
    {
      int err = OB_SUCCESS;
      UNUSED(log_task);
      TBSYS_LOG(ERROR, "ObUpsLogMgr::handle_unexpected()");
      err = OB_ERR_UNEXPECTED;
      return err;
    }

    int ObUpsLogMgr:: handle_write(ObBatchLogEntryTask* log_task)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = write_log_(log_task->log_data_, log_task->log_data_len_)))
      {
        TBSYS_LOG(ERROR, "write_log()=>%d", err);
      }
      return err;
    }

    int ObUpsLogMgr:: handle_response_client(ObBatchLogEntryTask* log_task)
    {
      int err = OB_SUCCESS;
      TBSYS_LOG(DEBUG, "handle_response_client(min_id=%ld, max_id=%ld, n_log_tasks=%ld)",
                log_task->min_id_, log_task->max_id_, log_task->n_log_tasks_);
      if (!log_task->is_valid())
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "log_task->is_valid()=>false");
      }
      else
      {
        int resp_err = OB_SUCCESS;
        for (int64_t i = 0; OB_SUCCESS == err && i < log_task->n_log_tasks_; i++)
        {
          ObLogEntryTask cur_log_task = log_task->log_tasks_[i];
          if (OB_SUCCESS != (resp_err = response_client_(cur_log_task.err_, cur_log_task.conn_, cur_log_task.channel_id_)))
          {
            TBSYS_LOG(WARN, "response_result()=>%d", resp_err);
          }
        }
      }
      return err;
    }

    int ObUpsLogMgr:: handle_response_master(ObBatchLogEntryTask* log_task)
    {
      int err = OB_SUCCESS;
      TBSYS_LOG(DEBUG, "handle_response_master(min_id=%ld, max_id=%ld, n_log_tasks=%ld)",
                log_task->min_id_, log_task->max_id_, log_task->n_log_tasks_);
      if (!log_task->is_valid())
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "log_task->is_valid()=>false");
      }
      else if (log_task->n_log_tasks_ <= 0)
      {
        TBSYS_LOG(ERROR, "log_task->n_log_task <= 0");
      }
      else
      {
        int resp_err = OB_SUCCESS;
        ObLogEntryTask cur_log_task = log_task->log_tasks_[0];
        if (OB_SUCCESS != (resp_err = response_client_(cur_log_task.err_, cur_log_task.conn_, cur_log_task.channel_id_)))
        {
          TBSYS_LOG(WARN, "response_result()=>%d", resp_err);
        }
      }
      return err;
    }

    int ObUpsLogMgr:: handle_send_log(ObBatchLogEntryTask* log_task)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = send_log_(log_task->log_data_, log_task->log_data_len_)))
      {
        if (OB_PARTIAL_FAILED != err)
        {
          TBSYS_LOG(ERROR, "send_log()=>%d", err);
        }
        else
        {
          TBSYS_LOG(WARN, "send_log()=>%d", err);
        }
      }
      if (OB_PARTIAL_FAILED == err)
      {
        err = OB_SUCCESS;
      }
      return err;
    }

    int ObUpsLogMgr:: apply_batch(const char* log_data, int64_t data_len)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = replay_log_in_buf_func(_stop, log_data, data_len, log_handler_)))
      {
        TBSYS_LOG(ERROR, "replay_log_in_buf_func(log_data=%p, data_len=%ld)=>%d", log_data, data_len, err);
      }
      return err;
    }

    int ObUpsLogMgr:: handle_apply(ObBatchLogEntryTask* log_task)
    {
      int err = OB_SUCCESS;
      char* log_data = log_task->log_data_;
      int64_t data_len = log_task->log_data_len_;
      if (OB_SUCCESS != (err = apply_batch(log_data, data_len)))
      {
        TBSYS_LOG(ERROR, "replay_log_in_buf_func(log_data=%p, data_len=%ld)=>%d", log_data, data_len, err);
      }
      return err;
    }

    int ObUpsLogMgr::write_and_flush_log2(const common::LogCommand cmd, const char* log_data, const int64_t data_len,
                                       tbnet::Connection* conn, uint32_t channel_id)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (OB_SUCCESS != (err = fill_log(cmd, log_data, data_len, conn, channel_id)))
      {
        TBSYS_LOG(ERROR, "fill_log(cmd=%d, log_data=%p, data_len=%ld, conn=%p, channel_id=%u)=>%d", cmd, log_data, data_len, conn, channel_id, err);
      }
      else if (OB_SUCCESS != (err = submit_log()))
      {
        TBSYS_LOG(ERROR, "submit_log()=>%d", err);
      }
      else if (OB_SUCCESS != (err = flush_log()))
      {
        TBSYS_LOG(ERROR, "flush_log()=>%d", err);
      }
      return err;
    }

    int64_t ObUpsLogMgr::get_cur_log_seq() const
    {
      int64_t log_seq = OB_INVALID_ID;
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else
      {
        log_seq = log_cursor_.log_id_;
      }
      return log_seq;
    }

    int64_t ObUpsLogMgr::get_last_net_elapse() const
    {
      return 0;
    }

    int64_t ObUpsLogMgr::get_last_disk_elapse() const
    {
      return 0;
    }

    int ObUpsLogMgr::fetch_log(const ObServer master, const ObLogCursor& start_cursor, ObLogCursor& end_cursor, const int64_t timeout)
    {
      int err = OB_SUCCESS;
      char* log_data = NULL;
      int64_t log_len = 0;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "is_inited()=>false");
      }
      else if (OB_SUCCESS != (err = fetch_handler_->fetch_log(master, start_cursor, end_cursor, log_data, log_len, timeout)))
      {
        if(OB_RESPONSE_TIME_OUT != err && OB_NEED_RETRY != err && OB_NOT_REGISTERED != err)
        {
          TBSYS_LOG(ERROR, "fetch_log error, err=%d log_cursor_=%s, log_data=%p log_len=%ld",
                    err, start_cursor.to_str(), log_data, log_len);
        }
      }
      else if (log_len <= 0)
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "receive empty log");
      }
      else if (OB_SUCCESS != (err = write_log_((const char*)log_data, log_len)))
      {
        TBSYS_LOG(ERROR, "ObCommitLogReceiver write_data error, err=%d log_cursor=%s log_data=%p, data_len=%ld",
                  err, log_cursor_.to_str(), log_data, log_len);
      }
      else if (OB_SUCCESS != (err = apply_batch(log_data, log_len)))
      {
        TBSYS_LOG(ERROR, "apply_batch(log_data=%p, data_len=%ld)=>%d", log_data, log_len, err);
      }
      return err;
    }
    int ObUpsLogMgr:: start_fetch(const volatile bool& stop, const ObServer& master, const ObLogCursor& start_cursor, ObLogCursor& end_cursor, const int64_t timeout)
    {
      int err = OB_SUCCESS;
      end_cursor = start_cursor;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
        role_mgr_->set_state(ObRoleMgr::ERROR);
        TBSYS_LOG(ERROR, "ObUpsLogFetcher has not initialized");
      }
      while (!stop && !_stop && OB_SUCCESS == err)
      {
        err = fetch_log(master, end_cursor, end_cursor, timeout);
        //TBSYS_LOG(INFO, "fetch_log(cursor=%s)", log_cursor_.to_str());
        if (OB_SUCCESS == err)
        {
          log_cursor_ = end_cursor;
        }
        if (OB_SUCCESS != err && OB_RESPONSE_TIME_OUT != err && OB_NEED_RETRY != err && OB_NOT_REGISTERED != err)
        {
          TBSYS_LOG(ERROR, "fetch_log error, err=%d log_cursor_=%s", err, end_cursor.to_str());
        }
        else if (OB_RESPONSE_TIME_OUT == err || OB_NEED_RETRY == err)
        {
          err = OB_SUCCESS;
        }
        else if (OB_NOT_REGISTERED == err)
        {
          TBSYS_LOG(INFO, "The master returned NOT_REGISTERED, go to register");
          role_mgr_->set_state(ObRoleMgr::INIT);
          while (!_stop && ObRoleMgr::INIT == role_mgr_->get_state())
          {
            usleep(10000);
          }
          err = OB_SUCCESS;
        }
        else if (OB_SUCCESS != (err = enable(end_cursor.log_id_)) && OB_NEED_RETRY != err)
        {
          TBSYS_LOG(ERROR, "log_queue->enable(%s)=>%d", end_cursor.to_str(), err);
        }
        else if (OB_NEED_RETRY == err)
        {
          err = OB_SUCCESS;
        }
        else
        {
          TBSYS_LOG(INFO, "fetcher catch up receiver: %s", end_cursor.to_str());
          role_mgr_->set_state(ObRoleMgr::ACTIVE);
          break;
          while (ObRoleMgr::ACTIVE == role_mgr_->get_state())
          {
            usleep(10000);
          }
        }

        if (OB_SUCCESS != err)
        {
          role_mgr_->set_state(ObRoleMgr::ERROR);
        }
      }
      TBSYS_LOG(INFO, "ObUpsLogFetcher finished, err=%d role_state=%s", err, role_mgr_->get_state_str());
      return err;
    }

    ObUpsLogMgr::LogReplayRunnable::LogReplayRunnable(): log_mgr_(NULL), finished_(false), return_code_(0), min_log_file_id_(-1), max_log_file_id_(-1)
    {
    }

    ObUpsLogMgr::LogReplayRunnable::~LogReplayRunnable()
    {}

    int ObUpsLogMgr::LogReplayRunnable::init(ObUpsLogMgr* log_mgr, const int64_t min_log_file_id, const int64_t max_log_file_id)
    {
      finished_ = false;
      return_code_ = OB_SUCCESS;
      log_mgr_ = log_mgr;
      min_log_file_id_ = min_log_file_id;
      max_log_file_id_ = max_log_file_id;
      return OB_SUCCESS;
    }

    void ObUpsLogMgr::LogReplayRunnable::start()
    {
      CDefaultRunnable::start();
    }

    void ObUpsLogMgr::LogReplayRunnable:: run(tbsys::CThread* thread, void* arg)
    {
      UNUSED(thread);
      UNUSED(arg);
      TBSYS_LOG(INFO, "replay_thread([%ld, %ld])", min_log_file_id_, max_log_file_id_);
      if(log_mgr_)
      {
        return_code_ = log_mgr_->replay_this_range(min_log_file_id_, max_log_file_id_);
        finished_ = true;
      }
    }

    int ObUpsLogMgr::LogReplayRunnable:: wait_replay_finished(bool& stop, int64_t retry_wait_time_us)
    {
      while(!stop && !finished_)
      {
        usleep(retry_wait_time_us);
      }
      return return_code_;
    }

    ObUpsLogMgr::LogFetchRunnable::LogFetchRunnable(): log_mgr_(NULL), master_(), start_cursor_(), timeout_(-1)
    {
    }

    ObUpsLogMgr::LogFetchRunnable::~LogFetchRunnable()
    {
    }

    int ObUpsLogMgr::LogFetchRunnable::init(ObUpsLogMgr* log_mgr, const common::ObServer &master, const ObLogCursor& log_cursor, const int64_t timeout)
    {
      int err = OB_SUCCESS;

      if (NULL == log_mgr || !log_cursor.is_valid())
      {
        TBSYS_LOG(ERROR, "Parameters are invalid in ObUpsLogMgr:LogFetcher::init, log_mgr=%p, cursot=%s", log_mgr, log_cursor.to_str());
        err = OB_INVALID_ARGUMENT;
      }
      else
      {
        master_ = master;
        log_mgr_ = log_mgr;
        start_cursor_ = log_cursor;
        timeout_ = timeout;
      }
      return err;
    }
    void ObUpsLogMgr::LogFetchRunnable::destroy_threads()
    {
      if (_thread) {
        delete[] _thread;
        _thread = NULL;
      }
      _stop = false;
    }

    int ObUpsLogMgr::LogFetchRunnable::reset()
    {
      int err = OB_SUCCESS;
      if (false) // (!is_initialized_) // 不检查
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "ObUpsLogFetcher.reset()=>%d", err);
      }
      else
      {
        stop();
        wait();
        destroy_threads();
      }
      return err;
    }

    void ObUpsLogMgr::LogFetchRunnable::run(tbsys::CThread* thread, void* arg)
    {
      int err = OB_SUCCESS;
      UNUSED(thread);
      UNUSED(arg);
      ObLogCursor end_cursor;
      if (NULL == log_mgr_ || !start_cursor_.is_valid())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "ObUpsLogMgr:LogFetcher::run(log_mgr=%p, cursor=%s)=>%d", log_mgr_, start_cursor_.to_str(), err);
      }
      else if (OB_SUCCESS != (err = log_mgr_->start_fetch(_stop, master_, start_cursor_, end_cursor, timeout_)))
      {
        TBSYS_LOG(ERROR, "log_mgr->start_fetch(log_cursor=%s, timeout=>%ld)=>%d", start_cursor_.to_str(), timeout_, err);
      }
    }
  } // end namespace updateserver
} // end namespace oceanbase
