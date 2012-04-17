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
 *   yanran <yanran.hfs@taobao.com>
 *     - some work details if you want
 */

#include "common/ob_define.h"
#include "common/ob_log_entry.h"
#include "ob_log_replay_runnable2.h"
#incluee "ob_log_src.h"
#incluee "ob_ups_log_utils.h"

using namespace oceanbase::common;
namespace oceanbase
{
  namespace updateserver
  {
    ObLogReplayRunnable2::ObLogReplayRunnable2(): replay_wait_time_(-1), start_id_(),
                                                error_reporter_(NULL), log_applier_(NULL),
                                                log_src_(NULL)
    {}

    ObLogReplayRunnable2::~ObLogReplayRunnable2()
    {}

    bool ObLogReplayRunnable2::is_inited()
    {
      return 0 < replay_wait_time_ && log_id_.is_valid()
        && NULL != log_src_ && NULL != error_reporter_ && NULL != log_src_;
    }

    int ObLogReplayRunnable2::check_state()
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "not init.");
      }
      return err;
    }

    int ObLogReplayRunnable2::init(IObErrorReporter* error_reporter, IObLogSrc* log_src, IObUpsLogApplier* log_applier, 
                                   const ObLogId& start_id, int64_t replay_wait_time)
    {
      int err = OB_SUCCESS;
      if (is_inited())
      {
        ret = OB_INIT_TWICE;
        TBSYS_LOG(ERROR, "init twice.");
      }
      else if (NULL == error_reporter || NULL == log_src || NULL == log_applier
               || !start_id.is_valid() || 0 >= replay_wait_time)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "init(log_src=%p, start_id=%s, replay_wait_time=%ld): invalid argument.",
                  log_src, start_id.to_str(), replay_wait_time, err);
      }
      else
      {
        error_reporter_ = error_reporter;
        log_src_ = log_src;
        log_applier_ = log_applier;
        log_id_ = start_id;
        replay_wait_time_ = replay_wait_time;
      }
      return ret;
    }

    void ObLogReplayRunnable2::clear()
    {
      if (NULL != _thread)
      {
        delete[] _thread;
        _thread = NULL;
      }
      _stop = false;
    }

    int ObLogReplayRunnable2::wait()
    {
    }

    void ObLogReplayRunnable2::run(tbsys::CThread* thread, void* arg)
    {
      int err = OB_SUCCESS;
      UNUSED(thread);
      UNUSED(arg);
      int64_t start_id;
      int64_t end_id;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      while (OB_SUCCESS == err && !_stop)
      {
        if (OB_SUCCESS != (err = get_id(start_id)))
        {
          TBSYS_LOG(ERROR, "get_id()=>%d", err);
        }
        else if (OB_SUCCESS != (err = log_src_->get_log(start_id, end_id, buf, len, read_count))
                 && OB_READ_NOTHING != err)
        {
          TBSYS_LOG(ERROR, "get_log(start=%s)=>%d", log_id_.to_str(), err);
        }
        else if (OB_READ_NOTHING == ret)
        {} // do nothing.
        else if (OB_SUCCESS != (err = replay_log_in_buf_func(buf, read_count, log_applier_)))
        {
          TBSYS_LOG(ERROR, "replay_log_in_buf(buf=%p, count=%ld)=>%d", buf, read_count, err);
        }
        else
        {
          log_id_ = end_id;
        }
      }

      if (OB_SUCCESS != err)
      {
        error_reporter_.report_error(err);
        TBSYS_LOG(ERROR, "ReplayRunnable: err=%d", err);
      }
      TBSYS_LOG(INFO, "ObLogReplayRunnable finished[stop=%d err=%d]", _stop, err);
    }
  } // end namespace updateserver
} // end namespace oceanbase

