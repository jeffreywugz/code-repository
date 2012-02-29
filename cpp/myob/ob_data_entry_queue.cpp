/**
 * (C) 2007-2010 Alibaba Group Holding Limited.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * version 2 as published by the Free Software Foundation.
 *
 * Version: $Id$
 *
 * ob_data_buffer_queue.cpp
 *
 * Authors:
 *   yuanqi.xhf <yuanqi.xhf@taobao.com>
 *
 */

#include "ob_data_entry_queue.h"
#include "tblog.h"
#include "ob_atomic.h"
#include "common/ob_malloc.h"
#include "Time.h"

using namespace oceanbase::common;

namespace oceanbase
{
  namespace updateserver
  {
    int ObDataEntry::init(int64_t id, int64_t len, char* buf, int32_t n_reader)
    {
      int err = OB_SUCCESS;
      if (NULL == buf || len <= 0)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "init(id=%ld, len=%ld, buf=%ld)=>%d", id, len, buf, err);
      }
      else
      {
        magic_ = OB_DATA_ENTRY_MAGIC;
        buf_size_ = len;
        id_ = id;
        n_reader_ = n_reader;
        buf_ = buf;
      }
      return err;
    }

    ObDataEntry* ObDataEntry::get_next() const
    {
      return (ObDataEntry*)(this->buf_ + this->buf_size_);
    }

    int64_t ObDataEntry:: full_data_len() const
    {
      return sizeof(*this) + buf_size_;
    }

    int ObDataEntry:: commit()
    {
      atomic_dec((uint32_t*)&n_reader_);
      if(n_reader_ < 0)
      {
        TBSYS_LOG(ERROR, "ObDataEntry(%p).n_reader == %ld", this, n_reader_);
      }
      return n_reader_;
    }

    int ObDataEntry:: dump_for_debug() const
    {
      int err = OB_SUCCESS;
      if (!is_valid())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "ObDataEntry(%p).is_valid(magic=%lx, id=%ld, size=%ld, n_reader=%d, buf=%p)=>false",
                  this, magic_, id_, buf_size_, n_reader_, buf_);
      }
      else
      {
        fprintf(stderr, "ObDataEntry{magic=%lx, id=%ld, size=%ld, n_reader=%d, buf=%p}\n",
                magic_, id_, buf_size_, n_reader_, buf_);
      }
      return err;
    }

    int ObDataEntry::get_data_entry(char* buf, ObDataEntry*& entry)
    {
      int err = OB_SUCCESS;
      ObDataEntry* entry_ = NULL;
      if (NULL == buf)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "NULL == buf");
      }
      else
      {
        entry_ = (ObDataEntry*)(buf - sizeof(ObDataEntry));
        if(!entry_->is_valid())
        {
          err = OB_ERR_UNEXPECTED;
          TBSYS_LOG(ERROR, "entry->is_valid(magic=%x, buf_size_=%ld, id=%ld, n_reader=%ld)=>false",
                    entry_->magic_, entry_->buf_size_, entry_->id_, entry_->n_reader_);
        }
        else
        {
          entry = entry_;
        }
      }
      return err;
    }

    bool ObDataEntry:: is_valid() const
    {
      return NULL != this && magic_ == OB_DATA_ENTRY_MAGIC && buf_size_ > 0 && NULL != buf_ && id_ >= 0 && n_reader_ >= 0;
    }

    int ObDataEntry:: serialize(char* buf, int64_t len, int64_t& pos) const
    {
      int err = OB_SUCCESS;
      char* start = buf + pos;
      if (!is_valid())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "is_valid()=>false");
      }
      else if (NULL == buf || len <= 0)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "serialize(buf=%p, len=%ld, pos=%ld)=>%d",
                  buf, len, pos, err);
      }
      else if (pos + full_data_len() > len)
      {
        err = OB_BUF_NOT_ENOUGH;
        TBSYS_LOG(DEBUG, "serialize(buf=%p, len=%ld, pos=%ld)=>%d",
                  buf, len, pos, err);
      }
      else
      {
        memcpy(start, this, sizeof(*this));
        memcpy(start + sizeof(*this), this->buf_, this->buf_size_);
        ((ObDataEntry*)start)->buf_ = start + sizeof(*this);
        pos += full_data_len();
      }
      return err;
    }

    ObDataEntryBlock::ObDataEntryBlock(): next_(NULL), is_frozen_(false), buf_()
    {
      reset();
    }

    int ObDataEntryBlock:: init(ObDataEntryBlock* next, char* buf, int64_t capacity)
    {
      int err = OB_SUCCESS;
      if (NULL == next || NULL == buf || capacity <= 0)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "init(next=%p, buf=%p, capacity=%ld)=>%d", next, buf, capacity, err);
      }
      else
      {
        next_ = next;
        buf_.set_data(buf, capacity);
        reset();
      }
      return err;
    }

    int ObDataEntryBlock:: check_state() const
    {
      int err = OB_SUCCESS;
      if (NULL == buf_.get_data())
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "check_state(version=%ld, buf=%p, capacity=%ld)=>%d", version_, buf_.get_data(), buf_.get_capacity(), err);
      }
      return err;
    }

    void ObDataEntryBlock::reset()
    {
      version_ = -1;
      buf_.get_position() = 0;
      is_frozen_ = false;
      min_id_ = -1;
      max_id_ = -1;
    }

    int64_t ObDataEntryBlock::get_version() const
    {
      return version_;
    }

    int64_t ObDataEntryBlock:: get_max_id() const
    {
      return max_id_;
    }

    int64_t ObDataEntryBlock:: get_min_id() const
    {
      return min_id_;
    }

    bool ObDataEntryBlock::hold(int64_t id) const
    {
      int64_t cur_max_id = max_id_;
      int64_t cur_min_id = min_id_;
      return id == 0 || (cur_min_id >= 0 && cur_max_id >= cur_min_id) && (id >= cur_min_id && id <= cur_max_id);
    }

    char* ObDataEntryBlock::to_str() const
    {
      snprintf(str_buf_, sizeof(str_buf_),
               "ObDataEntryBlock{this=%p, version=%ld, min_id=%ld, max_id=%ld, buf=%p, capacity=%ld, len=%ld}",
               this, version_, min_id_, max_id_, buf_.get_data(), buf_.get_capacity(), buf_.get_position());
      str_buf_[sizeof(str_buf_)-1] = 0;
      return str_buf_;
    }

    int ObDataEntryBlock::dump_for_debug() const
    {
      int err = OB_SUCCESS;
      const ObDataEntry* entry = NULL;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "ObDataEntryBlock(%p).check_state()=>%d", this, err);
      }
      else
      {
        fprintf(stderr, "%s\n", to_str());
        for(entry = (ObDataEntry*)buf_.get_data(); can_read_((char*)entry); entry = entry->get_next())
        {
          if(OB_SUCCESS != (err = entry->dump_for_debug()))
          {
            TBSYS_LOG(ERROR, "ObDataEntryBlock(%p).dump_for_debug()=>%d", this, err);
            break;
          }
        }
      }
      return err;
    }

    int ObDataEntryBlock::dump_block_ring_for_debug() const
    {
      int err = OB_SUCCESS;
      const ObDataEntryBlock* target = this;
      while(OB_SUCCESS == err)
      {
        if(NULL == target)
        {
          err = OB_ERR_UNEXPECTED;
          TBSYS_LOG(ERROR, "dump_block_ring_for_debug(block == NULL)");
        }
        else if(OB_SUCCESS != (err = target->dump_for_debug()))
        {
          TBSYS_LOG(ERROR, "target[%p]->dump_block_ring_for_debug()=>%d", target, err);
        }
        else if(this == (target = target->next_))
        {
          break;
        }
      }
      return err;
    }

    int ObDataEntryBlock:: seek_in_block_ring(int64_t id, ObDataEntryBlock*& target_block, ObDataEntry*& target_entry, ObDataEntry* start)
    {
      int err = OB_SUCCESS;
      TBSYS_LOG(DEBUG, "seek_in_block_ring(id=%ld, start=%p)", id, start);
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if(id < 0)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "check_argument(id=%ld, start=%p)=>%d", id, start, err);
      }
      else if (OB_SUCCESS != (err = select_block(id, target_block)))
      {
        TBSYS_LOG(WARN, "select_block(id=%ld)=>%d", id, err);
      }
      else if(OB_SUCCESS != (err = target_block->seek(id, target_entry, start)))
      {
        TBSYS_LOG(WARN, "target_block->get(id=%ld)=>%d", id, err);
      }
      return err;
    }

    int ObDataEntryBlock:: get_from_block_ring(int64_t last_id, ObDataEntryBlock*& target_block, ObDataEntry*& target_entry,
                                               ObDataEntry* last_entry)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if(NULL != last_entry && (OB_SUCCESS != (err = get_next(target_block, target_entry, last_entry))))
      {
        TBSYS_LOG(ERROR, "get_next(last_id=%ld)=>%d", last_id, err);
      }
      else if(NULL == last_entry && (OB_SUCCESS != (err = seek_in_block_ring(last_id+1, target_block, target_entry, NULL))))
      {
        TBSYS_LOG(ERROR, "seek_in_block_ring(last_id=%ld)=>%d", last_id, err);
      }
      return err;
    }

    int ObDataEntryBlock:: push_to_block_ring(int64_t max_id_processed, ObDataEntry* entry, ObDataEntryBlock*& target_block)
    {
      int err = OB_SUCCESS;
      int64_t version = version_;
      ObDataEntryBlock* tmp_target_block = this;
      if(-1 == version)
      {
        version = 0;
      }
      if (NULL == entry)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "push_to_block_ring(entry=NULL)=>%d", err);
      }
      else
      {
        err = push(version, max_id_processed, entry);
      }

      if(OB_SUCCESS != err && OB_BUF_NOT_ENOUGH != err)
      {
        TBSYS_LOG(ERROR, "block(%p).push(version=%ld, id=%ld, max_id=%ld, max_id_processed=%ld)=>%d",
                  tmp_target_block, version, entry->id_, max_id_, max_id_processed, err);
      }
      else if (OB_BUF_NOT_ENOUGH == err)
      {
        tmp_target_block = tmp_target_block->next_;
        if(NULL == tmp_target_block)
        {
          err = OB_ERR_UNEXPECTED;
          TBSYS_LOG(ERROR, "next == NULL", this);
        }
        else if(OB_SUCCESS != (err = tmp_target_block->push(version+1, max_id_processed, entry)))
        {
          if (OB_BUF_NOT_ENOUGH == err)
          {
            TBSYS_LOG(WARN, "block->next(%p).push(version=%ld, id=%ld, max_id=%ld, max_id_processed=%ld)=>%d",
                      tmp_target_block, version+1, entry->id_, tmp_target_block->max_id_, max_id_processed, err);
          }
          else
          {
            TBSYS_LOG(ERROR, "block->next(%p).push(version=%ld, id=%ld, max_id=%ld, max_id_processed=%ld)=>%d",
                      tmp_target_block, version+1, entry->id_, tmp_target_block->max_id_, max_id_processed, err);
          }
        }
      }
      if(OB_SUCCESS == err)
      {
        target_block = tmp_target_block;
      }
      return err;
    }

    int ObDataEntryBlock:: select_block(int64_t id, ObDataEntryBlock*& target)
    {
      int err = OB_SUCCESS;
      ObDataEntryBlock* p = this;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if(id < 0)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "check_argument(id=%ld)=>%d", id, err);
      }
      else
      {
        while(true)
        {
          if(NULL == p)
          {
            TBSYS_LOG(ERROR, "select_block(): NULL == p");
          }
          if(p->hold(id))
          {
            target = p;
            break;
          }
          if(this == (p = p->next_))
          {
            err = OB_DATA_NOT_SERVE;
            break;
          }
        }
      }
      if (OB_SUCCESS != err)
      {
        TBSYS_LOG(ERROR, "select_block(id=%ld, min_id=%ld, max_id=%ld, version=%ld)=>%d",
                  id, min_id_, max_id_, version_, err);
      }
      return err;
    }

    int ObDataEntryBlock:: push(int64_t version, int64_t max_id_processed, ObDataEntry* entry)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if(OB_SUCCESS != (err = reclaim(max_id_processed)))
      {
        if(OB_REF_NUM_NOT_ZERO != err)
        {
          TBSYS_LOG(ERROR, "reclaim(this_version=%ld, target_version=%ld, max_id=%ld, max_id_processed=%ld)=>%d",
                    version_, version, max_id_, max_id_processed, err);
        }
        else
        {
          err = OB_BUF_NOT_ENOUGH;
          TBSYS_LOG(WARN, "reclaim(this_version=%ld, target_version=%ld, max_id=%ld, max_id_processed=%ld)=>%d",
                    version_, version, max_id_, max_id_processed, err);
        }
      }
      else if(OB_SUCCESS != (err = push_(entry)))
      {
        TBSYS_LOG(DEBUG, "push_(id=%ld, len=%ld, n_reader_=%d, data=%p)=>%d",
                  entry->id_, entry->buf_size_, entry->n_reader_, entry->buf_, err);
      }
      else
      {
        if(version_ != version)
        {
          TBSYS_LOG(INFO, "block(%p, id=%ld, max_id_processed=%ld).version: %ld=>%ld", this,
                    entry->id_, max_id_processed, version_, version);
        }
        version_ = version;
      }
      return err;
    }

    int ObDataEntryBlock:: seek(int64_t id, ObDataEntry*& target, ObDataEntry* start)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if(!can_read_((char*)start) && OB_SUCCESS != (err = seek_(id, target)))
      {
        TBSYS_LOG(ERROR, "seek_(id=%ld)=>%d", id, err);
      }
      else if(!can_read_((char*)target))
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "!can_read_(this=%p, id=%ld, target=%p, start=%p)", this, id, target, start);
      }
      else if(target->id_ < id)
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "target->id(%ld) != id(%ld)", target->id_, id);
      }
      return err;
    }

    int ObDataEntryBlock:: seek_(int64_t id, ObDataEntry*& target)
    {
      int err = OB_ENTRY_NOT_EXIST;
      ObDataEntry* entry = NULL;
      for(entry = (ObDataEntry*)buf_.get_data(); can_read_((char*)entry); entry = entry->get_next())
      {
        if(entry->id_ >= id)
        {
          target = entry;
          err = OB_SUCCESS;
          break;
        }
      }
      if (OB_SUCCESS != err)
      {
        TBSYS_LOG(ERROR, "seek(id=%ld, min_id=%ld, max_id=%ld, buf_=%p, len=%ld, capacity=%ld)=>%d",
                  id, min_id_, max_id_, buf_.get_data(), buf_.get_position(), buf_.get_capacity(), err);
      }
      else if(!target->is_valid())
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "!target->is_valid(id=%ld, target=%p)", id, target);
      }
      else if(target->id_ < id)
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "target->id_(%ld) != id(%ld)", target->id_, id);
      }
      return err;
    }

    int ObDataEntryBlock:: get_next(ObDataEntryBlock*& target_block, ObDataEntry*& target_entry, ObDataEntry* cur_entry)
    {
      int err = OB_SUCCESS;
      err = this->get_next_may_fail(target_block, target_entry, cur_entry);
      if (OB_SUCCESS != err && OB_DATA_NOT_SERVE != err)
      {
        TBSYS_LOG(ERROR, "get_next_may_fail()=>%d", err);
      }
      else if (OB_DATA_NOT_SERVE == err && version_ + 1 == this->next_->version_
               && OB_SUCCESS != (err = this->next_->get_next_may_fail(target_block, target_entry, NULL)))
      { //switch buf
        TBSYS_LOG(ERROR, "get_next_may_fail()=>%d", err);
      }
      return err;
    }

    int ObDataEntryBlock:: get_next_may_fail(ObDataEntryBlock*& target_block, ObDataEntry*& target_entry, ObDataEntry* cur_entry)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(WARN, "get_next():check_state()=>%d", err);
      }
      else if(NULL == cur_entry)
      {
        target_entry = (ObDataEntry*)buf_.get_data();
      }
      else
      {
        target_entry = cur_entry->get_next();
      }

      if (OB_SUCCESS != err)
      {}
      else if (!can_read_((char*)target_entry))
      {
        err = OB_DATA_NOT_SERVE;
        TBSYS_LOG(DEBUG, "get_next():target_block->can_read(target_entry=%p)=>false", target_entry);
      }
      if (OB_SUCCESS == err)
      {
        target_block = this;
      }
      return err;
    }

    int ObDataEntryBlock::reclaim(int64_t max_id_processed)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if(is_frozen_)
      {
         //Note: 为了处理SWITCH_LOG， max_id_ == max_id_processed时不能释放
        if(max_id_ != -1 && max_id_processed != -1 && max_id_ >= max_id_processed)
        {
          err = OB_REF_NUM_NOT_ZERO;
          TBSYS_LOG(WARN, "reclaim(this=%p, version=%ld, max_id=%ld, max_id_processed=%ld)=>%d",
                    this, version_, max_id_, max_id_processed, err);
        }
        else
        {
          TBSYS_LOG(DEBUG, "reclaim(this=%p, version=%ld, max_id=%ld, max_id_processed=%ld)=>%d",
                    this, version_, max_id_, max_id_processed, err);
          reset();
        }
      }
      return err;
    }

    // 检查block能否读以start开始的entry
    bool ObDataEntryBlock:: can_read_(char* start) const
    {
      return ! (start < buf_.get_data() || start + sizeof(ObDataEntry) >= buf_.get_data() + buf_.get_position());
    }

    int ObDataEntryBlock::push_(ObDataEntry* entry)
    {
      int err = OB_SUCCESS;
      if(is_frozen_)
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "is_frozen_ == true");
      }
      else if(!entry->is_valid())
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "check_argument(id=%ld, len=%ld, buf=%p, pos=%ld, capacity=%ld)=>%d",
                  entry->id_, entry->buf_size_, entry->buf_, buf_.get_position(), buf_.get_capacity(), err);
      }
      else if(OB_BUF_NOT_ENOUGH == (err = entry->serialize(buf_.get_data(), buf_.get_capacity(), buf_.get_position())))
      {
        is_frozen_ = true;
        TBSYS_LOG(DEBUG, "buf_not_enough(id=%ld, min_id=%ld, max_id=%ld, len=%ld, buf=%p, pos=%ld, capacity=%ld)",
                  entry->id_, min_id_, max_id_, entry->buf_size_, entry->buf_, buf_.get_position(), buf_.get_capacity());
      }
      else if (OB_SUCCESS == err)
      {
        max_id_ = entry->id_;
        if(min_id_ == -1)
        {
          min_id_ = entry->id_;
        }
      }
      return err;
    }


    ObDataEntryIterator:: ObDataEntryIterator(): queue_(NULL), last_id_(-1), last_block_(NULL), cur_block_(NULL),
                                                 last_entry_(NULL), cur_entry_(NULL)
    {
    }

    int ObDataEntryIterator:: init(ObDataEntryQueue* queue)
    {
      int err = OB_SUCCESS;
      if (NULL == queue)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "init(queue=%p)=>%d", err);
      }
      else
      {
        queue_ = queue;
      }
      return err;
    }

    int ObDataEntryIterator:: reset()
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else
      {
        last_id_ = -1;
        last_block_ = NULL;
        cur_block_ = NULL;
        last_entry_ = NULL;
        cur_entry_ = NULL;
      }
      return err;
    }

    int ObDataEntryIterator:: check_state()
    {
      int err = OB_SUCCESS;
      if (NULL == queue_)
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      return err;
    }

    int ObDataEntryIterator:: commit(ObDataEntry* entry)
    {
      int err = OB_SUCCESS;
      int n_reader = 0;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (NULL == entry || cur_entry_ != entry)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(WARN, "invalid_argument(entry=%p, cur_entry_=%p)", entry, cur_entry_);
      }
      else if(0 > (n_reader = entry->commit()))
      {
        err = OB_ERR_UNEXPECTED;
        TBSYS_LOG(ERROR, "n_reader(%d) < 0", n_reader);
      }
      else if(0 == n_reader)
      {
        queue_->set_max_id_processed(entry->id_);
      }
      last_id_ = cur_id_;
      last_entry_ = cur_entry_;
      last_block_ = cur_block_;
      return err;
    }

    int ObDataEntryIterator:: commit_by_buf(char* buf)
    {
      int err = OB_SUCCESS;
      ObDataEntry* entry = NULL;
      if(OB_SUCCESS != (err = ObDataEntry::get_data_entry(buf, entry)))
      {
        TBSYS_LOG(ERROR, "get_data_entry(buf=%p)=>%d", buf, err);
      }
      else if(OB_SUCCESS != (err = commit(entry)))
      {
        TBSYS_LOG(ERROR, "entry.commit(entry=%p)=>%d", entry, err);
      }
      return err;
    }

    int ObDataEntryIterator::  get(ObDataEntry*& entry, int64_t timeout)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if(OB_SUCCESS != (err = queue_->get(last_id_, cur_block_, cur_entry_, last_block_, last_entry_, timeout)))
      {
        if(OB_NEED_RETRY == err)
        {
          TBSYS_LOG(DEBUG, "queue->get(last_id=%ld, entry=>%p)=>%d", last_id_, entry, err);
        }
        else
        {
          TBSYS_LOG(ERROR, "queue->get(last_id=%ld, entry=>%p)=>%d", last_id_, entry, err);
        }
      }
      else
      {
        entry = cur_entry_;
        cur_id_ = entry->id_;
      }
      return err;
    }

    int ObDataEntryIterator:: get_buf(char*& buf, int64_t timeout)
    {
      int err = OB_SUCCESS;
      ObDataEntry* entry = NULL;
      if(OB_SUCCESS != (err = get(entry, timeout)))
      {
        if (OB_NEED_RETRY != err)
        {
          TBSYS_LOG(ERROR, "get(entry=%p, timeout=%ld)=>%d", entry, timeout, err);
        }
      }
      else
      {
        buf = entry->buf_;
      }
      return err;
    }

    ObDataEntryQueue:: ObDataEntryQueue(): buf_(NULL), n_iters_(0), max_id_processed_(-1), max_id_pushed_(-1), max_id_can_hold_(-1),
                                           block_size_(0), cur_write_buf_(NULL)
    {
    }

    ObDataEntryQueue:: ~ObDataEntryQueue()
    {
        if (NULL != buf_)
        {
          ob_free(buf_);
          buf_ = NULL;
        }
    }
    
    int ObDataEntryQueue:: check_state() const
    {
      int err = OB_SUCCESS;
      if(n_iters_ < 0 || 0 == block_size_ || NULL == cur_write_buf_)
      {
        err = OB_NOT_INIT;
        TBSYS_LOG(ERROR, "ObDataEntryQueue.check_state(n_iters_=%ld, block_size=%ld, cur_write_buf=%p)=>%d",
                  n_iters_, block_size_, cur_write_buf_, err);
      }
      return err;
    }

    int ObDataEntryQueue:: init(int64_t block_size, int n_reader, ObDataEntryIterator*& iters, int64_t retry_wait_time)
    {
      int err = OB_SUCCESS;
      char* buf = NULL;
      if(block_size <= 0 || n_reader <= 0 || n_reader > (int)ARRAYSIZEOF(iterators_))
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "ObDataEntryQueue.init(block_size=%ld, n_reader=%ld)=>%d", block_size, n_reader, err);
      }
      else if (NULL != buf_)
      {
        err = OB_INIT_TWICE;
        TBSYS_LOG(ERROR, "ObDataEntryQueue.init(block_size=%ld, n_reader=%ld)=>%d", block_size, n_reader, err);
      }
      else
      {
        n_iters_ = n_reader;
        block_size_ = block_size;
        retry_wait_time_ = retry_wait_time;
        if (NULL == (buf = (char*)ob_malloc(block_size*2)))
        {
          err = OB_ALLOCATE_MEMORY_FAILED;
          TBSYS_LOG(ERROR, "ob_malloc(size=%ld)=>NULL", block_size*2);
        }
      }
      if (OB_SUCCESS == err)
      {
        for(int i = 0; i < 2; i++)
        {
          if(OB_SUCCESS != (err = block_[i].init(&block_[1-i], buf + i*block_size, block_size)))
          {
            TBSYS_LOG(ERROR, "block[%d].init(buf=%ld, block_size=%ld)=>%d", i, buf, block_size, err);
            break;
          }
        }
      }
      if (OB_SUCCESS == err)
      {
        for(int i = 0; i < n_reader; i++)
        {
          if(OB_SUCCESS != (err = iterators_[i].init(this)))
          {
            TBSYS_LOG(ERROR, "iterators[%d].init()=>%d", i, err);
            break;
          }
        }
      }
      if (OB_SUCCESS == err)
      {
        cur_write_buf_ = &block_[0];
        iters = iterators_;
        if(OB_SUCCESS != (err = reset()))
        {
          TBSYS_LOG(ERROR, "push(id=0)", err);
        }
      }
      return err;
    }

    int ObDataEntryQueue:: push_sentinel()
    {
      int err = OB_SUCCESS;
      static char sentinel[] = "SENTINEL";
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if(OB_SUCCESS != (err = push(0, sizeof(sentinel), sentinel, 0)))
      {
        TBSYS_LOG(ERROR, "push(id=0)", err);
      }
      return err;
    }

    int ObDataEntryQueue:: reset()
    {
      int err = OB_SUCCESS;
      int64_t max_id = 0;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (OB_SUCCESS != (err = flush(max_id)))
      {
        TBSYS_LOG(ERROR, "flush()=>%d", err);
      }
      else
      {
        max_id_processed_ = -1;
        max_id_pushed_ = -1;
        max_id_can_hold_ = -1;
        cur_write_buf_ = &block_[0];
        block_[0].reset();
        block_[1].reset();
        for(int i = 0; i < n_iters_; i++)
        {
          if(OB_SUCCESS != (err = iterators_[i].reset()))
          {
            TBSYS_LOG(ERROR, "iterators[%d].reset()=>%d", i, err);
            break;
          }
        }
      }
      return err;
    }

    int ObDataEntryQueue:: clear()
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = reset()))
      {
        TBSYS_LOG(ERROR, "reset()=>%d", err);
      }
      else
      {
        if (NULL != buf_)
        {
          ob_free(buf_);
          buf_ = NULL;
        }
      }
      return err;
    }
    
    char* ObDataEntryQueue::to_str()
    {
      snprintf(str_buf_, sizeof(str_buf_),
               "ObDataEntryQueue{n_iters=%d, max_id_processed=%ld, max_id_can_hold=%ld, block[0]=%s, block[1]=%s}",
               n_iters_, max_id_processed_, max_id_can_hold_, block_[0].to_str(), block_[1].to_str());
      str_buf_[sizeof(str_buf_)-1] = 0;
      return str_buf_;
    }

    int ObDataEntryQueue::dump_for_debug() const
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "ObDataEntryBlock(%p).check_state()=>%d", this, err);
      }
      else
      {
        cur_write_buf_->dump_block_ring_for_debug();
      }
      return err;
    }

    int ObDataEntryQueue:: push_may_need_retry(ObDataEntry* entry)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if(max_id_can_hold_ != -1 && entry->id_ > max_id_can_hold_)
      {
        err = OB_SIZE_OVERFLOW;
        TBSYS_LOG(WARN, "push(id=%ld, len=%ld, buf=%p, max_id_can_hold=%ld)=>%d",
                  entry->id_, entry->buf_size_, entry->buf_, max_id_can_hold_, err);
      }
      else if(OB_SUCCESS != (err = cur_write_buf_->push_to_block_ring(max_id_processed_, entry, cur_write_buf_)))
      {
        TBSYS_LOG(WARN, "cur_write_buf.push_to_block_ring(id=%ld, len=%ld, buf=%p)=>%d", entry->id_, entry->buf_size_, entry->buf_, err);
      }
      else
      {
        max_id_pushed_ = entry->id_;
      }
      if(OB_BUF_NOT_ENOUGH == err)
      {
        err = OB_NEED_RETRY;
      }
      return err;
    }

    int ObDataEntryQueue::  push(int64_t id, int64_t len, char* buf, int64_t timeout)
    {
      int err = OB_SUCCESS;
      tbutil::Time now = tbutil::Time::now();
      ObDataEntry entry;
      if (OB_SUCCESS != (err = entry.init(id, len, buf, n_iters_)))
      {
        TBSYS_LOG(ERROR, "entry.init(id=%ld, len=%ld, buf=%p)=>%d", id, len, buf, err);
      }
      else
      {
        while(true)
        {
          err = push_may_need_retry(&entry);
          if (OB_SUCCESS == err)
          {
            break; // success cause exit
          }
          else if(OB_NEED_RETRY != err)
          {
            TBSYS_LOG(ERROR, "push(id=%ld, len=%ld, buf=%p)=>%d", id, len, buf, err);
            break; // error cause exit
          }
          else if (timeout >= 0 && (tbutil::Time::now() - now).toMilliSeconds() > timeout/1000)
          {
            break; // timeout cause exit
          }
          usleep(retry_wait_time_);
        }
      }
      return err;
    }

    int ObDataEntryQueue:: enable(int64_t max_id)
    {
      int err = OB_SUCCESS;
      int64_t old_version = -1;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if(max_id < 0)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "enable(max_id[%ld] < 0)=>%d", max_id, err);
      }
      else if(-1 != max_id_processed_)
      {
        err = OB_INIT_TWICE;
        TBSYS_LOG(ERROR, "ObDataEntry.enable(max_id=%ld)=>%d", max_id, err);
      }
      else
      {
        old_version = cur_write_buf_->get_version();
        if (max_id == 0 && old_version == -1 && OB_SUCCESS != (err = push_sentinel()))
        {
          TBSYS_LOG(ERROR, "cur_write_buf_->push_sentinel()");
        }
        else if(!cur_write_buf_->hold(max_id))
        {
          err = OB_NEED_RETRY;
          TBSYS_LOG(DEBUG, "ObDataEntry.enable(max_id=%ld)=>%d", max_id, err);
        }
      }
      if (OB_SUCCESS == err)
      {
        max_id_processed_ = max_id;
        if(cur_write_buf_->get_version() > old_version + 1)
        {
          err = OB_ERR_UNEXPECTED;
          TBSYS_LOG(ERROR, "ObDataEntry.enable(max_id=%ld)=>%d", max_id, err);
        }
      }
      return err;
    }

    int ObDataEntryQueue:: flush(int64_t& max_id)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if (max_id_processed_ == -1)
      {
        TBSYS_LOG(WARN, "flush():max_id_processed==-1");
      }
      else
      {
        while(true)
        {
          if(OB_NEED_RETRY != (err = flush_may_need_retry(max_id)))
          {
            break;
          }
        }
      }
      return err;
    }

    int ObDataEntryQueue:: get_may_need_retry(int64_t last_id, ObDataEntryBlock*& target_block, ObDataEntry*& target_entry,
                                              ObDataEntryBlock* last_block, ObDataEntry* last_entry)
    {
      int err = OB_SUCCESS;
      if (-1 == last_id)
      {
        last_id = max_id_processed_;
      }
      if (NULL == last_block)
      {
        last_block = cur_write_buf_;
      }
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else if(-1 != max_id_processed_ && last_id < max_id_processed_)
      {
        err = OB_DATA_NOT_SERVE;
        TBSYS_LOG(DEBUG, "ObDataEntryQueue.get(id=%ld, max_id_processed=%ld, max_id_can_hold=%ld)=>%d",
                  last_id, max_id_processed_, max_id_can_hold_, err);
      }
      else if(-1 == max_id_processed_ || (-1 != max_id_can_hold_ && last_id >= max_id_can_hold_))
      {
        err = OB_NEED_RETRY;
      }
      else if (last_id < 0)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "invalid_argument(id=%ld)", last_id);
      }
      else if(last_id >= max_id_pushed_)
      {
        err = OB_NEED_RETRY;
        TBSYS_LOG(DEBUG, "get_may_need_retry(last_id[%ld] >= max_id_pushed[%ld])=>%d", last_id, max_id_pushed_, err);
      }
      else if(OB_SUCCESS != (err = last_block->get_from_block_ring(last_id, target_block, target_entry, last_entry)))
      {
        TBSYS_LOG(ERROR, "get_from_block_ring(id=%ld)=>%d", last_id, err);
      }
      return err;
    }

    int ObDataEntryQueue:: get(int64_t last_id, ObDataEntryBlock*& target_block, ObDataEntry*& target_entry,
                               ObDataEntryBlock* last_block, ObDataEntry* last_entry, int64_t timeout)
    {
      int err = OB_SUCCESS;
      tbutil::Time now = tbutil::Time::now();

      while(true)
      {
        err = get_may_need_retry(last_id, target_block, target_entry, last_block, last_entry);
        if (OB_SUCCESS == err)
        {
          break; // success cause exit
        }
        else if(OB_NEED_RETRY != err)
        {
          TBSYS_LOG(ERROR, "get()=>%d", err);
          break; // error cause exit
        }
        else if (timeout >= 0 && (tbutil::Time::now() - now).toMilliSeconds() > timeout/1000)
        {
          break; // timeout cause exit
        }
        usleep(retry_wait_time_);
      }
      return err;
    }

    int ObDataEntryQueue:: flush_may_need_retry(int64_t& max_id)
    {
      int err = OB_SUCCESS;
      if (OB_SUCCESS != (err = check_state()))
      {
        TBSYS_LOG(ERROR, "check_state()=>%d", err);
      }
      else
      {
        max_id = cur_write_buf_->get_max_id();
        disable_(max_id);
        wait_();
        if(cur_write_buf_->get_max_id() != max_id)
        {
          err = OB_NEED_RETRY;
        }
      }
      return err;
    }

    void ObDataEntryQueue:: set_max_id_processed(int64_t max_id)
    {
      max_id_processed_ = max_id;
    }

    void ObDataEntryQueue:: set_max_id_can_hold(int64_t max_id)
    {
      max_id_can_hold_ = max_id;
    }

    void ObDataEntryQueue:: wait_()
    {
      while(max_id_processed_ < max_id_can_hold_)
      {
        usleep(retry_wait_time_);
      }
    }

    void ObDataEntryQueue:: disable_(int64_t max_id)
    {
      max_id_can_hold_ = max_id;
    }

  } // end namespace updateserver
} // end namespace oceanbase

