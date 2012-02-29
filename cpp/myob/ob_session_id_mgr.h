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

#ifndef __OCEANBASE_UPDATESERVER_OB_SESSION_ID_MGR_H__
#define __OCEANBASE_UPDATESERVER_OB_SESSION_ID_MGR_H__

namespace oceanbase
{
  namespace updateserver
  {
    class ObSessionIdMgr
    {
      struct Session
      {
        Session(): session_id_(0), last_access_time_us_(0) {}
        ~Session(){}
        int64_t session_id_;
        int64_t last_access_time_us_;
      };
      public:
        ObSessionIdMgr();
        virtual ~ObSessionIdMgr();
        int init(const int64_t n_cached_session, const int64_t session_live_duration_us);
        int get_capacity(int64_t& n_session) const;
        int acquire_session(const int64_t advised_session_id, int64_t& returned_session_id);
        int releas_session(const int64_t session_id);
      protected:        
        bool is_inited() const;
        int check_state() const;
      private:
        volatile int64_t max_session_id_;
        int64_t n_cached_session_;
        int64_t session_live_duration_us_;
        Session* sessions_;
    };
  }; // end namespace updateserver
}; // end namespace oceanbase
#endif // __OCEANBASE_UPDATESERVER_OB_SESSION_ID_MGR_H__
