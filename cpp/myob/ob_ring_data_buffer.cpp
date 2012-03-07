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
#include "tbsys.h"
#include "common/utility.h"
#include "common/ob_malloc.h"
#include "ob_ring_data_buffer.h"
#include "ob_data_block.h"

using namespace oceanbase::common;

namespace oceanbase
{
  namespace updateserver
  {
    // x != 0
    bool is_power_of_2(int64_t x)
    {
      return 0 == (x & (x-1));
    }

    ObRingDataBuffer::ObRingDataBuffer(): data_buf_(NULL), block_size_shift_(0),
                                          n_blocks_(0), start_pos_(0), end_pos_(0)
    {}

    ObRingDataBuffer::~ObRingDataBuffer()
    {
      free_buf();
    }

    void ObRingDataBuffer::free_buf()
    {
      if (NULL != data_buf_)
      {
        ob_free(data_buf_);
        data_buf_ = NULL;
      }
    }

    bool ObRingDataBuffer::is_inited() const
    {
      return NULL != data_buf_ && NULL != blocks_ && 0 <= n_blocks_ && 0 <= block_size_shift_;
    }

    int ObRingDataBuffer::dump_for_debug() const
    {
      int err = OB_SUCCESS;
      TBSYS_LOG(INFO, "pos=[%ld,%ld)", start_pos_, end_pos_);
      TBSYS_LOG(INFO, "block=%ld*(1<<%ld)", n_blocks_, block_size_shift_);
      if (NULL != blocks_)
      {
        for(int i = 0; i < n_blocks_; i++)
        {
          blocks_[i].dump_for_debug();
        }
      }
      return err;
    }

    int ObRingDataBuffer::check_state() const
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      return err;
    }

    int64_t ObRingDataBuffer::get_next_block_pos(const int64_t pos) const
    {
      return ((pos >> block_size_shift_) + 1) << block_size_shift_;
    }

    int ObRingDataBuffer::init(const int64_t n_blocks, const int64_t block_size_shift)
    {
      int err = OB_SUCCESS;
      if (is_inited())
      {
        err = OB_INIT_TWICE;
      }
      else if (n_blocks <= 0 || n_blocks > MAX_N_BLOCKS || 0 > block_size_shift)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else if (NULL == (data_buf_ = (char*)ob_malloc(n_blocks * 1<<block_size_shift)))
      {
        err = OB_ALLOCATE_MEMORY_FAILED;
      }
      for (int64_t i = 0; OB_SUCCESS == err && i < n_blocks; i++)
      {
        if (OB_SUCCESS != (err = blocks_[i].init(data_buf_ + i * (1<<block_size_shift), block_size_shift)))
        {
          TBSYS_LOG(ERROR, "blocks[%ld].init()=>%d", i, err);
        }
      }
      if (OB_SUCCESS != err)
      {
        free_buf();
      }
      else
      {
        block_size_shift_ = block_size_shift;
        n_blocks_ = n_blocks;
      }
      return err;
    }

    int64_t ObRingDataBuffer::get_start_pos() const
    {
      return start_pos_;
    }

    int64_t ObRingDataBuffer::get_end_pos() const
    {
      return end_pos_;
    }

    bool ObRingDataBuffer::is_pos_valid(const int64_t pos) const
    {
      return pos >= start_pos_ && pos < end_pos_;
    }

    int ObRingDataBuffer::read(const int64_t pos, char* buf, const int64_t len, int64_t& read_count) const
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (NULL == buf || 0 >= len || 0 > pos)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "read(buf=%p[%ld], pos):invalid argument", buf, len, pos);
      }
      else if (!is_pos_valid(pos))
      {
        err = OB_DATA_NOT_SERVE;
      }
      else if (OB_SUCCESS != (err = blocks_[(pos>>block_size_shift_) % n_blocks_].read(pos, buf, len, read_count)))
      {
        if (OB_DATA_NOT_SERVE != err && OB_FOR_PADDING != err)
        {
          TBSYS_LOG(ERROR, "block[%ld].read(pos=%ld)=>%d", pos>>block_size_shift_, pos, err);
        }
      }
      else if (!is_pos_valid(pos))
      {
        err = OB_DATA_NOT_SERVE;
      }
      return err;
    }

    int ObRingDataBuffer::next_block_for_append(const int64_t pos, int64_t& new_pos)
    {
      int err = OB_SUCCESS;
      ObDataBlock* cur_block = NULL;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (pos < 0 || pos != end_pos_)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else
      {
        new_pos = get_next_block_pos(pos);
        end_pos_ = new_pos;
        cur_block = blocks_ + ((new_pos>>block_size_shift_) % n_blocks_);
        start_pos_ = max(start_pos_, get_next_block_pos(cur_block->get_end_pos()));
      }

      if (OB_SUCCESS == err && OB_SUCCESS != (err = cur_block->reset(end_pos_)))
      {
        TBSYS_LOG(ERROR, "block[%ld].reset(new_pos=%ld)=>%d", end_pos_>>block_size_shift_, end_pos_, err);
      }
      return err;
    }

    int ObRingDataBuffer::append(const int64_t pos, const char* buf, const int64_t len)
    {
      int err = OB_SUCCESS;
      int64_t new_pos = pos + len;
      int64_t next_block_err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (NULL == buf || 0 > len || 0 > pos || len > (1<<block_size_shift_))
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "append(buf=%p[%ld], pos):invalid argument", buf, len, pos);
      }
      else if (pos != end_pos_)
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "pos[%ld] != end_pos_[%ld]", pos, end_pos_);
      }
      else if (OB_SUCCESS != (err = blocks_[(pos>>block_size_shift_) % n_blocks_].append(pos, buf, len))
               && OB_BUF_NOT_ENOUGH != err)
      {
        TBSYS_LOG(ERROR, "blocks[%ld].append(pos=%ld)=>%d", pos>>block_size_shift_, pos, err);
      }
      else if (OB_BUF_NOT_ENOUGH == err && OB_SUCCESS != (next_block_err = next_block_for_append(pos, new_pos)))
      {
        err = next_block_err;
        TBSYS_LOG(ERROR, "next_block_for_append(pos=%ld)=%d", pos, next_block_err);
      }
      else
      {
        end_pos_ = new_pos;
      }
      return err;
    }
  }; // end namespace updateserver
}; // end namespace oceanbase


