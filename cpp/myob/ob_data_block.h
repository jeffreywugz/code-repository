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
#ifndef __OB_UPDATESERVER_OB_DATA_BLOCK_H__
#define __OB_UPDATESERVER_OB_DATA_BLOCK_H__
#include "common/ob_define.h"

namespace oceanbase
{
  namespace updateserver
  {
    class ObDataBlock
    {
      public:
        ObDataBlock();
        ~ObDataBlock();
      public:
        int init(char* block_buf, const int64_t block_size_shift);
        // read()之前和之后都要检查pos是否有效，有效指pos在[start_pos_, end_pos_) 区间之内。
        // 这个read()函数不负责判断读取数据的有效性，数据的有效性由调用者检查
        // 读取的长度为 min(end_pos_-pos, len)
        // read可能返回: OB_DATA_NOT_SERVE表示pos无效
        //   如果当前block已经frozen，并且pos == end_pos_, 返回OB_FOR_PADDING, read_count设置为结尾剩余空间的大小
        int read(const int64_t pos, char* buf, const int64_t len, int64_t& read_count) const;
        // append的数据长度超过了buf_size, 返回OB_ERR_UNEXPECTED;
        // append之前要检查pos是否与end_pos_相等，如果不等，认为这是错误;
        // append一批数据之前如果剩余空间不够，把当前block的frozen设置为true，返回OB_BUF_NOT_ENOUGH，
          // 向一个已经frozen的block追加数据，也会返回OB_BUF_NOT_ENOUGH
        int append(const int64_t pos, const char* buf, const int64_t len);
        // reset()设置start_pos_和end_pos_的值为pos， pos的值必须大于end_pos_
        int reset(const int64_t pos);
      public:
        bool is_frozen() const;
        int64_t get_block_size() const;
        int64_t get_start_pos() const;
        int64_t get_end_pos() const;
        int dump_for_debug() const;
      protected:
        bool is_inited() const;
        bool is_pos_valid(const int64_t pos, const int64_t end_pos) const;
        int64_t get_offset(const int64_t pos) const;
        int64_t get_start_pos(const int64_t pos) const;
      private:
        DISALLOW_COPY_AND_ASSIGN(ObDataBlock);
        char* block_buf_;
        int64_t block_size_shift_;
        // start_pos_, end_pos_是指在多个data block组成的buf中的全局位置
        // start_pos_对应着这个block的开始位置, 可以把start_pos_看作这个block的version
        // ObDataBlock不记录start_pos_, 只记录end_pos_, 要求buf_size_只能是2的n次幂: buf_size_ = 1<<buf_size_shift_.
        // start_pos_可以计算出来: start_pos_ = end_pos_ & ((1<<buf_size_shift_)-1))
        // volatile int64_t start_pos_;
        volatile int64_t end_pos_;
        volatile int64_t frozen_pos_; // frozen之后end_pos_的值， 未frozen时frozen_pos_ == -1
    };
  }; // end namespace updateserver
}; // end namespace oceanbase
#endif /* __OB_UPDATESERVER_OB_DATA_BLOCK_H__ */
