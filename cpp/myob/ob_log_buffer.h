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
#ifndef __OB_UPDATESERVER_OB_FIXED_RING_BUFFER_H__
#define __OB_UPDATESERVER_OB_FIXED_RING_BUFFER_H__
#include "common/ob_define.h"

namespace oceanbase
{
  namespace updateserver
  {
    class ObFixedRingBuffer
    {
      public:
        ObFixedRingBuffer();
        ~ObFixedRingBuffer();
        // read() append() reset() 全部可以多线程安全地被调用
        // append()/reset()被强制串行化
        // read()成功，可以保证读取的一定是最后一次reset()之后放入缓冲区的数据，
        // 并且读取期间没有再次调用reset()。
        int init(const int64_t buf_len, char* buf);
        // last_reset_pos_, start_pos_, end_pos_, next_end_pos_ 只会递增，折回到实际缓冲区开始也不重置为0，
        // 如果缓冲区无限大，[last_reset_pos_, end_pos) 之间的数据有效，不过实际缓冲区有限，
        // 有效数据范围是 [max(start_pos_, end_pos-buf_len_), end_pos)
        //
        // next_end_pos_在实际append数据之前被修改为追加数据之后end_pos_应该有的值，
        // 实际append数据之后再修改end_pos_的值， 这就意味着实际append()数据期间next_end_pos_ > end_pos_
        //
        // reset()操作也类似，先改next_end_pos_, 最后再修改end_pos_:
        //    next_end_pos_++; start_pos_ = next_end_pos_; end_pos_ = next_end_pos_;
        //
        // 并发append()和reset()操作执行前用CAS指令争用next_end_pos_的修改权，这样保证互斥。

        // 为保证read()返回成功时，读到的数据一定有效，在读取之前和读取之后都要检查读取的起始点pos是否有效，
        // 有效指pos在 [max(start_pos_, end_pos-buf_len_), end_pos) 区间之内。
        // 为了保险，可以检验读取开始之前和结束之后，start_pos_的值是否发生了变化，如果发生了变化，
        // 表明读取期间调用了reset()，读取的数据一定无效。
        // 不过实际上如果start_pos_发生了变化，start_pos_的值，一定大于pos的值，所以，通不过上面的区间检查。
        int reset();
        // 允许len == 0
        int read(const int64_t pos, char* buf, const int64_t len, int64_t& read_count) const;
        // 允许len == 0
        int append(const int64_t pos, const int64_t start_id, const int64_t end_id, const char* buf, const int64_t len);

      public:
        int64_t get_start_pos() const;
        int64_t get_end_pos() const;
        int64_t get_next_end_pos() const;

      protected:
        bool is_pos_valid(const int64_t pos) const;
        bool is_inited() const;
        int check_state() const;
      protected:
        volatile int64_t last_reset_pos_;
        volatile int64_t start_pos_;
        volatile int64_t end_pos_;
        volatile int64_t next_end_pos_;
        int64_t start_id;
        int64_t end_id;
        int64_t buf_len_;
        char* buf_;
      private:
        DISALLOW_COPY_AND_ASSIGN(ObFixedRingBuffer);
    };
  }; // end namespace updateserver
}; // end namespace oceanbase

#endif /* __OB_UPDATESERVER_OB_FIXED_RING_BUFFER_H__ */
