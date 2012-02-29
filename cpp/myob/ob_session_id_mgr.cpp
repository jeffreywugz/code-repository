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

#include "common/ob_define.h"
#include "Time.h"

using namespace oceanbase::common;
namespace oceanbase
{
  namespace updateserver
  {
    ObSessionIdMgr::ObSessionIdMgr(): max_session_id_(0), n_cached_session_(0), session_live_duration_us_(0), sessions_(NULL)
    {}

    ObSessionIdMgr::~ObSessionIdMgr()
    {
      if (NULL != sessions_)
      {
        delete []sessions_;
      }
    }

    bool ObSessionIdMgr::is_inited() const
    {
      return NULL != sessions_ && n_cached_session_ > 0 && session_live_duration_us_ > 0;
    }

    int ObSessionIdMgr::check_state() const
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      return err;
    }

    int ObSessionIdMgr::get_capacity(int64_t& n_session) const
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else
      {
        n_session = n_cached_session_;
      }
      return err;
    }

    int ObSessionIdMgr::init(const int64_t n_cached_session, const int64_t session_live_duration_us)
    {
      int err = OB_SUCCESS;
      if (is_inited())
      {
        err = OB_INIT_TWICE;
        TBSYS_LOG(ERROR, "init twice.");
      }
      else if (0 >= n_cached_session || 0 >= session_live_duration_us)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "init(n_cached_session=%ld, session_duration=%ld): invalid argument",
                  n_cached_session, session_live_duration_us);
      }
      else if (NULL == (sessions_ = new(std::nothrow) Session[n_cached_session]))
      {
        err = OB_ALLOCATE_MEMORY_FAILED ;
        TBSYS_LOG(ERROR, "new Session[%ld] failed.", n_cached_session);
      }

      if (OB_SUCCESS == err)
      {
        n_cached_session_ = n_cached_session_;
        session_live_duration_us_ = session_live_duration_us;
      }
      return err;
    }
      
    int ObSessionIdMgr::acquire_session(const int64_t advised_session_id, int64_t& returned_session_id)
    {
      int err = OB_ENTRY_NOT_EXIST;
      int candinate_id = (0 >= advised_session_id?  __sync_add_and_fetch(&max_session_id_, 1): advised_session_id);
      Session* session = NULL;
      int64_t now = tbutil::Time::now().toMicroSeconds();

      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      for(int i = 0; OB_SUCCESS != err && i <= n_cached_session_; i++)
      {
        session = sessions_ + candinate_id%n_cached_session_;
        if(__sync_bool_compare_and_swap(&session->session_id_, 0, candinate_id))
        {
          if (session->last_access_time_us_ + session_live_duration_us_ < now)
          {
            session->last_access_time_us_ = now;
            err = OB_SUCCESS;
            break;
          }
          else if (OB_SUCCESS != (err = release_session(candinate_id)))
          {
            TBSYS_LOG(ERROR, "release_session(candinate_id=%ld)=>%d", candinate_id, err);
          }
        }
        candinate_id = __sync_add_and_fetch(&max_session_id_, 1);
      }
      if (OB_SUCCESS == err)
      {
        returned_session_id = candinate_id;
      }
      return err;
    }

    int ObSessionIdMgr::releas_session(const int64_t session_id)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if(__sync_bool_compare_and_swap(&sessions_[session_id%n_cached_session_].session_id_, session_id, 0))
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "release_session(session_id=%ld): I don't hold this session", session_id);
      }
      return err;
    }
  } // end namespace updateserver
} // end namespace oceanbase
