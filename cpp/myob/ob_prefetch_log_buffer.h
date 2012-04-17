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
#ifndef __OB_UPDATESERVER_OB_PREFETCH_LOG_BUFFER_H__
#define __OB_UPDATESERVER_OB_PREFETCH_LOG_BUFFER_H__

#include "ob_log_src.h"
#include "ob_fixed_ring_buffer.h"

namespace oceanbase
{
  namespace updateserver
  {
    // ObPrefetchLogBuffer在备机重放日志时实现取日志和重放并行化，
    // 当replay线程每次调用get_log()获取日志检测到prefetch_log_buffer有剩余空间时，便提交一个异步取日志任务。
    // 异步取日志任务执行时从主机取回日志，调用push_log()将日志追加到prefetch_log_buffer中
    // 所以prefetch_log_buffer是一个单读者，多写者的环形缓冲区
    // 读者读取日志时检测到日志不连续，需要调用reset()重置prefetch_log_buffer.
    // 这意味着get_log()和reset()的调用一定是串行的，但是它们和push_log()的调用是并发的。

    // 读者读取位置由read_pos_记录，写者最后追加的位置由end_pos_记录,
    // read_pos_/end_pos_不断递增，遇到缓冲区末尾折回时不重置为0，
    // 每次读取和追加数据不能超过缓冲区长度。
    // 为了保证互斥，push_log()/reset()调用时要争用next_end_pos_的修改权。

    // append_log()时保证不会覆盖read_pos_, get_log()时不对数据有效性做检查，
    // get_log()保证读到的是完整的日志，但是有可能这些日志不是期望读到的日志，
    // 所以可能返回OB_DISCONTINUOUS_LOG
    class ObPrefetchLogBuffer : public IObLogSrc, public ObFixedRingBuffer
    {
      public:
        ObPrefetchLogBuffer();
        virtual ~ObPrefetchLogBuffer();
        // get_log()检测到要读的日志不连续时，返回OB_DISCONTINUOUS_LOG,
        virtual int get_log(const int64_t start_id, int64_t& end_id,
                    char* buf, const int64_t len, int64_t& read_count);
        // 追加的日志必须保证与缓冲区中已有的日志连续,
        // pos参数用作校验，目的是保证追加日志的操作被串行化(pos不对，append_log()不会成功)
        // push_log()可能返回OB_EAGAIN
        int append_log(const int64_t pos, const int64_t start_id, const int64_t end_id, const char* buf, const int64_t len);
        // reset()和get_log()串行地被调用
        int reset(const int64_t end_id);
      public:
        int dump_for_debug() const;
      private:
        DISALLOW_COPY_AND_ASSIGN(ObPrefetchLogBuffer);
        volatile int64_t end_id_;
        volatile int64_t read_pos_;
        volatile int64_t next_end_pos_; // 多个写者追加日志时，要争用next_end_pos_的修改权。
    };
  }; // end namespace updateserver
}; // end namespace oceanbase
#endif /* __OB_UPDATESERVER_OB_PREFETCH_LOG_BUFFER_H__ */
