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
#ifndef __OB_UPDATESERVER_OB_RING_DATA_BUFFER_H__
#define __OB_UPDATESERVER_OB_RING_DATA_BUFFER_H__
#include "common/ob_define.h"
#include "ob_data_block.h"
namespace oceanbase
{
  namespace updateserver
  {
    class ObDataBlock;
    // 不关心保存的数据是什么格式，
    // 不考虑并发append()
    // 有效数据的范围是 [start_pos_, end_pos_)
    // start_pos_, end_pos_只会递增，不会递减。
    class ObRingDataBuffer
    {
      static const int64_t MAX_N_BLOCKS = 128;
      static const int64_t DEFAULT_N_BLOCKS = 8;
      static const int64_t DEFAULT_BLOCK_SIZE_SHIFT = 25;
      public:
        ObRingDataBuffer();
        ~ObRingDataBuffer();
      public:
        // block_size必须是2的n次幂 1<<block_size_shift
        int init(const int64_t n_blocks = DEFAULT_N_BLOCKS,
                 const int64_t block_size_shift = DEFAULT_BLOCK_SIZE_SHIFT);
        // 可能返回OB_FOR_PADDING表示这次读取的数据无效，返回的read_count表示这次需要跳过的字节数
        // 可能返回OB_DATA_NOT_SERVE表示pos无效
        int read(const int64_t pos, char* buf, const int64_t len, int64_t& read_count) const;
        // 可能返回OB_BUF_NOT_ENOUGH: 表示当前block剩余空间不够写当前数据或之前就已经被froze
        int append(const int64_t pos, const char* buf, const int64_t len);
        int64_t get_next_block_pos(const int64_t pos) const;
        int64_t get_start_pos() const;
        int64_t get_end_pos() const;
        int dump_for_debug() const;
      protected:
        int next_block_for_append(const int64_t pos, int64_t& new_pos);
        bool is_pos_valid(const int64_t pos) const;
        bool is_inited() const;
        int check_state() const;
        void free_buf(); // init中途失败需要释放buf
      protected:
        DISALLOW_COPY_AND_ASSIGN(ObRingDataBuffer);
        char* data_buf_;
        int64_t block_size_shift_;
        int64_t n_blocks_;
        ObDataBlock blocks_[MAX_N_BLOCKS];
        int64_t start_pos_;
        int64_t end_pos_;
    };
  }; // end namespace updateserver
}; // end namespace oceanbase

#endif /* __OB_UPDATESERVER_OB_RING_DATA_BUFFER_H__ */
