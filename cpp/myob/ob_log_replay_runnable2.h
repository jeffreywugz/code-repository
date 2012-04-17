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
 *   yuanqi.xhf <yuanqi.xhf@taobao.com>
 *     - some work details if you want
 */

#ifndef OCEANBASE_COMMON_OB_LOG_REPLAY_RUNNABLE2_H_
#define OCEANBASE_COMMON_OB_LOG_REPLAY_RUNNABLE2_H_

#include "common/ob_log_entry.h"

namespace oceanbase
{
  namespace updateserver
  {
    class IObLogSrc;
#if 0                                           \
    //使用方法:
    ObLogReplayRunnable2 replay_runnable;
    replay_runnable.init(log_src, start, replay_wait_time);
    replay_runnable.start();
    // 停下线程
    replay_runnable.stop();
    replay_runnable.wait();
    replay_runnable.clear();
    // 获得log_id
    int64_t log_id = replay_runnable.get_log_id();
#endif
    class ObLogReplayRunnable2: public tbsys::CDefaultRunnable
    {
      public:
        ObLogReplayRunnable2();
        virtual ~ObLogReplayRunnable2();
        virtual int init(IObErrorReporter* error_reporter, IObLogSrc* log_src, IObUpsLogApplier* log_applier, 
                         int64_t start_id, int64_t replay_wait_time);
        virtual void run(tbsys::CThread* thread, void* arg);
        virtual void clear();
      protected:
        int64_t replay_wait_time_;
        int64_t next_id_;
        IObErrorReporter* error_reporter_;        
        IObUpsLogApplier* log_applier_;        
        IObLogSrc* log_src_;
    };
  } // end namespace updateserver
} // end namespace oceanbase
#endif // OCEANBASE_UPDATESERVER_OB_UPS_REPLAY_RUNNABLE2_H_

