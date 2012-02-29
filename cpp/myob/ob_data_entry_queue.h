/**
 * (C) 2007-2010 Alibaba Group Holding Limited.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * version 2 as published by the Free Software Foundation.
 *
 * Version: $Id$
 *
 * ob_data_buffer_queue.h
 *
 * Authors:
 *   yuanqi.xhf <yuanqi.xhf@taobao.com>
 *
 */
#ifndef __OCEANBASE_UPDATESERVER_OB_DATA_BUFFER_QUEUE_H__
#define __OCEANBASE_UPDATESERVER_OB_DATA_BUFFER_QUEUE_H__

#include "common/ob_define.h"
#include "common/data_buffer.h"

namespace oceanbase
{
  namespace updateserver
  {
    const int64_t OB_FIRST_ID_AVAILABLE = 0;
    const int64_t OB_DATA_ENTRY_MAGIC = 0x123456;
    const int64_t OB_MAX_N_ITERATORS = 3;
    const int OB_DATA_ENTRY_MAX_STR_LEN_FOR_INSPECT = (1<<10);

    // 每个ObDataEntry都带有一个唯一的id，
    // 在向ObDataEntryQueue中push()的时候要保证id严格递增，但是可以不连续
    struct ObDataEntry
    {
      int init(int64_t id, int64_t len, char* buf, int32_t n_reader);
      int64_t full_data_len() const;
      bool is_valid() const;
      static int get_data_entry(char* buf, ObDataEntry*& entry);

      int serialize(char* buf, int64_t len, int64_t& pos) const;
      ObDataEntry* get_next() const;
      int commit();
      int dump_for_debug() const;
      int64_t magic_;
      int64_t buf_size_;
      char* buf_;
      int64_t id_;
      int32_t n_reader_;
      ObDataEntry(): magic_(OB_DATA_ENTRY_MAGIC), buf_size_(0), buf_(NULL), id_(-1), n_reader_(0){}
      // 获得哨兵，作为第一个push()到队列中的项，它的id最小，
      // 保证在用id定位时，任何一个ObBatchLogEntryTask都有上一项。
      static ObDataEntry* get_sentinel()
      {
        static ObDataEntry entry;
        entry.init(0, 0, NULL, 0);
        return &entry;
      }

    };

    // 表示双缓冲区中的一个, 这个类的设计不限于一个block，可以把多个block连成一个环
    class ObDataEntryBlock
    {
      public:
        ObDataEntryBlock();
        int init(ObDataEntryBlock* next, char* buf, int64_t capacity);
        void reset();

        int check_state() const;

        // 由于id可以不连续，所以此函数只能判断给定block中是否包含有大于等于id的数据项,
        // 允许有另外的线程修改block，
        // 如果返回值为true，则可以确定在hold()函数调用的某个瞬间，给定id的数据项在当前block中
        bool hold(int64_t id) const;

        // 在block ring中查找编号大于等于id的项
        int seek_in_block_ring(int64_t id, ObDataEntryBlock*& target_block, ObDataEntry*& target_entry, ObDataEntry* start);

        // target_entry直接引用内部缓冲区, 在commit()之前，缓冲区有效
        // 从block_ring中返回cur_entry的下一项
        // err:
        //  + OB_SUCCESS:
        //  + OB_READ_NOTHING: 等待写者填充buf，应该重试
        //  + others: 错误，不应该重试
        int get_from_block_ring(int64_t last_id, ObDataEntryBlock*& target_block, ObDataEntry*& target_entry, ObDataEntry* last_entry);

        // 如果当前buffer已满，尝试使用下一个Buffer，只需要尝试当前buffer及下一个buffer。
        // err:
        //  + OB_SUCCESS:
        //  + OB_BUF_NOT_ENOUGH: 等待读者处理，应该重试
        //  + others: 错误，不应该重试
        int push_to_block_ring(int64_t max_id_processed, ObDataEntry* entry, ObDataEntryBlock*& target_block);

        int64_t get_version() const;
        int64_t get_max_id() const;
        int64_t get_min_id() const;

        char* to_str() const;
        int dump_for_debug() const;
        int dump_block_ring_for_debug() const;

      private:
        mutable char str_buf_[OB_DATA_ENTRY_MAX_STR_LEN_FOR_INSPECT];
        ObDataEntryBlock* next_;
        int64_t version_; // 表示是第几次使用block, 可用于统计和调试
        int64_t min_id_;
        int64_t max_id_;
        bool is_frozen_; // 等于NULL时，表示当前block被冻结
        common::ObDataBuffer buf_;

        //根据id找到所在的target, 在双buffer的情况下应该先尝试write_buffer, 后尝试read_buffer
        //select_block()会返回一个block，这个block在函数调用的某个瞬间，曾经包含了id对应的日志
        int select_block(int64_t id, ObDataEntryBlock*& target);

        // push()及push_to_block_ring()只会被一个写线程调用，
        // 读线程处理完之后会更新max_id_processed, 作为参数传递个push(),
        // push()通过这个值决定是否可以reclaim()老的block
        // 所以push()及push_to_block_ring()是安全的。
        int push(int64_t version, int64_t max_id_processed, ObDataEntry* entry);

        // 在启用iterator之前，iterator没有保存log的位置，所以此时需要seek到指定log的位置
        // start_point作为目标log起始地址的提示点，它是iterator计算出来的，
        // 在iterator第一次迭代时，start_point可以设置为NULL
        // 在当前找的log不在这个block时，即双buffer切换buffer时，start_point也是无效的。
        int seek(int64_t id, ObDataEntry*& target, ObDataEntry* start);
        int seek_(int64_t id, ObDataEntry*& pos);

        // 试图释放当前block
        int reclaim(int64_t max_id_processed);

        int get_next(ObDataEntryBlock*& target_block, ObDataEntry*& target_entry, ObDataEntry* cur_entry);
        int get_next_may_fail(ObDataEntryBlock*& target_block, ObDataEntry*& target_entry, ObDataEntry* cur_entry);
        int push_(ObDataEntry* entry);
        bool can_read_(char* start) const;
    };

    // ObDataEntryQueue的特性和限制:
    //  + 支持一个写者，多个读者， 但读者数目是在初始化时指定的
    //  + 任何一个数据块都有一个唯一ID，
    //  + 可以用ID控制ObDataEntryQueue中可以保存的数据块范围
    //    ObDataEntryQueue可以保存的数据项范围是[max_id_processed, max_id_can_hold]
    class ObDataEntryQueue;
    // 读者通过Iterator访问ObDataEntryQueue,
    // Iterator必须要在调用enable(max_id_processed)之后才可以访问。
    // 并且enable(max_id_processed)只能被成功地调用一次.
    class ObDataEntryIterator {
      public:
        ObDataEntryIterator();
        int init(ObDataEntryQueue* queue);
        int reset();
        int check_state();
        // 返回的buf直接引用内部缓冲区
        int get(ObDataEntry*& entry, int64_t timeout);
        int get_buf(char*& buf, int64_t timeout);
        int commit(ObDataEntry* entry);
        int commit_by_buf(char* buf);
      private:
        ObDataEntryQueue* queue_;
        int64_t last_id_;
        int64_t cur_id_;
        ObDataEntryBlock* last_block_;
        ObDataEntryBlock* cur_block_;
        ObDataEntry* last_entry_;
        ObDataEntry* cur_entry_;
    };

    class ObDataEntryQueue
    {
      friend class ObDataEntryQueueTest;
      public:
        ObDataEntryQueue();
        ~ObDataEntryQueue();
        int check_state() const;

        // block_size: 双Buffer的缓冲区大小
        int init(int64_t block_size, int n_reader, ObDataEntryIterator*& iters, int64_t retry_wait_time);
      public:        
        int clear();

        char* to_str();
        int dump_for_debug() const;

        // push(), get()的timeout参数单位为微妙
          //  + timeout > 0: 等待timeout时间
        //  + timout == 0: 不阻塞
        //  + timeout < 0: 无限期等待

         //只支持单线程调用push，
         // 拷贝数据到内部缓冲区, buf可以在函数返回后重用
         // err:
         //    + OB_SUCCESS:
         //    + OB_NEED_RETRY: 缓冲区已满, 应该重试
         //    + others: 错误，不应该重试
        int push(int64_t id, int64_t len, char* buf, int64_t timeout);

         // iterator会维护last_block, last_entry, last_id
         // 第一次调用时设置last_id = -1, last_block = NULL, last_entry = NULL
         // err:
         //    + OB_SUCCESS:
         //    + OB_NEED_RETRY: last_entry的下一项还未产生，应该重试
         //    + others: 错误，不应该重试
        int get(int64_t last_id, ObDataEntryBlock*& target_block, ObDataEntry*& target_entry,
                ObDataEntryBlock* last_block, ObDataEntry* last_entry, int64_t timeout);

        // 启用iter前，通过iter无法获得DataBufferEntry
        // err:
        //   + OB_SUCCESS:
        //   + OB_NEED_RETRY: max_id对应的数据项在函数返回前已经被覆盖了
        //   + others:
        int enable(int64_t max_id);

        // 禁止buf接受新日志，并等待buf中的日志都被处理完之后返回, 在备重新注册时有用
        int flush(int64_t& max_id);

        void set_max_id_processed(int64_t max_id);
        void set_max_id_can_hold(int64_t max_id);
      private:        
        int reset();
      private:
        int push_sentinel();
        int push_may_need_retry(ObDataEntry* entry);
        int get_may_need_retry(int64_t last_id, ObDataEntryBlock*& target_block, ObDataEntry*& target_entry,
                               ObDataEntryBlock* last_block, ObDataEntry* last_entry);
        int flush_may_need_retry(int64_t& max_id);
    private:
        char* buf_;        
        char str_buf_[OB_DATA_ENTRY_MAX_STR_LEN_FOR_INSPECT];
        ObDataEntryIterator iterators_[OB_MAX_N_ITERATORS];
        int n_iters_;
        int64_t max_id_processed_; // 小于等于此编号的日志都被处理完了
        int64_t max_id_pushed_; // 小于等于此编号的日志已经被push到缓冲区了
        int64_t max_id_can_hold_;  // 大于此编号的日志原则上不能放入缓冲区
        int64_t block_size_;
        ObDataEntryBlock block_[2];
        ObDataEntryBlock* cur_write_buf_;
        int64_t retry_wait_time_;


        void wait_();
        void disable_(int64_t max_id);
      };
    } // end namespace updateserver
  } // end namespace oceanbase
#endif // __OCEANBASE_UPDATESERVER_OB_DATA_BUFFER_QUEUE_H__
