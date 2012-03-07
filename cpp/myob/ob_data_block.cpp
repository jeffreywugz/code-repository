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
#include "ob_data_block.h"

using namespace oceanbase::common;

namespace oceanbase
{
  namespace updateserver
  {
    int64_t clear_lower_bits(const int64_t x, int64_t n_bits)
    {
      return x & ~((1<<n_bits)-1);
    }

    int64_t get_lower_bits(const int64_t x, int64_t n_bits)
    {
      return x & ((1<<n_bits)-1);
    }

    ObDataBlock::ObDataBlock(): block_buf_(NULL), block_size_shift_(0), end_pos_(0), frozen_pos_(-1)
    {}

    ObDataBlock::~ObDataBlock()
    {}

    bool ObDataBlock::is_inited() const
    {
      return NULL != block_buf_ && 0 < block_size_shift_;
    }

    int ObDataBlock::init(char* block_buf, const int64_t block_size_shift)
    {
      int err = OB_SUCCESS;
      if (is_inited())
      {
        err = OB_INIT_TWICE;
      }
      else if (NULL == block_buf || 0 >= block_size_shift)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else
      {
        block_buf_ = block_buf;
        block_size_shift_ = block_size_shift;
      }
      return err;
    }

    int ObDataBlock::dump_for_debug() const
    {
      int err = OB_SUCCESS;
      TBSYS_LOG(INFO, "blocks{frozen_pos=%ld, pos=[%ld, %ld)}", frozen_pos_,
                get_start_pos(), end_pos_);
      return err;
    }

    bool ObDataBlock::is_frozen() const
    {
      return 0 <= frozen_pos_;
    }

    int64_t ObDataBlock::get_block_size() const
    {
      return 1<<block_size_shift_;
    }

    int64_t ObDataBlock::get_start_pos(const int64_t pos) const
    {
      return clear_lower_bits(pos, block_size_shift_);
    }

    int64_t ObDataBlock::get_start_pos() const
    {
      return get_start_pos(end_pos_);
    }

    int64_t ObDataBlock::get_offset(const int64_t pos) const
    {
      return pos - get_start_pos(pos);
    }

    int64_t ObDataBlock::get_end_pos() const
    {
      return end_pos_;
    }

    int ObDataBlock::reset(const int64_t pos)
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (0 > pos)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else if (pos < end_pos_)
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "pos[%ld] < end_pos_[%ld]", pos, end_pos_);
      }
      else
      {
        end_pos_ = pos;
        frozen_pos_ = -1;
      }
      return err;
    }

    bool ObDataBlock::is_pos_valid(const int64_t pos, const int64_t end_pos) const
    {
      return pos >= get_start_pos(end_pos) && pos < end_pos;
    }

    int ObDataBlock::read(const int64_t pos, char* buf, const int64_t len, int64_t& read_count) const
    {
      int err = OB_SUCCESS;
      int64_t end_pos = end_pos_;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (NULL == buf || 0 >= len || 0 > pos)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "read(buf=%p[%ld], pos=%ld): invalid argument", buf, len, pos);
      }
      else if (pos == frozen_pos_)
      {
        err = OB_FOR_PADDING;
        read_count = get_start_pos(pos) + get_block_size() - pos;
      }
      else if (!is_pos_valid(pos, end_pos))
      {
        err = OB_DATA_NOT_SERVE;
      }
      else if (0 > (read_count = min(len, end_pos - pos))
               || NULL == memcpy(buf, block_buf_ + get_offset(pos), read_count))
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "data_block.read(pos=%ld, end_pos=%ld): unexpected error!", pos, end_pos_);
      }
      else if (!is_pos_valid(pos, end_pos_))
      {
        err = OB_DATA_NOT_SERVE;
      }
      return err;
    }

    int ObDataBlock::append(const int64_t pos, const char* buf, const int64_t len)
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (NULL == buf || 0 > len || 0 > pos || len > get_block_size())
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "append(buf=%p[%ld], pos=%ld): invalid argument", buf, len, pos);
      }
      else if (0 <= frozen_pos_)
      {
        err = OB_BUF_NOT_ENOUGH;
        TBSYS_LOG(WARN, "append to frozen buf: block=[%ld,%ld], buf=%p[%ld]",
                  get_start_pos(), end_pos_, buf, len);
      }
      else if (pos != end_pos_)
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "pos[%ld] != end_pos_[%ld]", pos, end_pos_);
      }
      else if (get_start_pos() + get_block_size()  < end_pos_ + len + 1)
        // 保证每个block最后都会至少预留1个字节，跳转到下一个block始终可以在这被检测到
      {
        err = OB_BUF_NOT_ENOUGH;
        frozen_pos_ = end_pos_;
        TBSYS_LOG(DEBUG, "data_block[%ld].frozen(pos=%ld)", end_pos_>>block_size_shift_, end_pos_);
      }
      else if (NULL == memcpy(block_buf_ + get_offset(end_pos_), buf, len))
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "data_block.append(pos=%ld, buf=%p[%ld]): unexpected error!", end_pos_, buf, len);
      }
      else
      {
        end_pos_ += len;
      }
      return err;
    }

  }; // end namespace updateserver
}; // end namespace oceanbase
