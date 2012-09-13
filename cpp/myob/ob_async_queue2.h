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
#ifndef __OB_COMMON_OB_ASYNC_QUEUE_H__
#define __OB_COMMON_OB_ASYNC_QUEUE_H__
#include "ob_malloc.h"
#include "pthread.h"

namespace oceanbase
{
  namespace common
  {
    struct ObCondPair
    {
      enum CondIdx
      {
        COND_ONE = 0,
        COND_TWO = 1,
      };
      timespec* make_abs_end_time(timespec* ts, int64_t timeout)
      {
        int64_t time_us = tbsys::CTimeUtil::getTime() + timeout;
        ts->tv_sec = time_us/1000000;
        ts->tv_nsec = (time_us%1000000) * 1000;
        return ts;
      }

      ObCondPair(): n_waiter1_(0), n_waiter2_(0)
      {
        pthread_mutex_init(&mutex_, NULL);
        pthread_cond_init(&cond1_, NULL);
        pthread_cond_init(&cond2_, NULL);
      }
      ~ObCondPair()
      {
        pthread_mutex_destroy(&mutex_);
        pthread_cond_destroy(&cond1_);
        pthread_cond_destroy(&cond2_);
      }
      int lock()
      {
        return pthread_mutex_lock(&mutex_);
      }
      int unlock()
      {
        return pthread_mutex_unlock(&mutex_);
      }
      int wait(const CondIdx idx, const int64_t timeout_us)
      {
        int err = OB_SUCCESS;
        timespec end_time;
        if (idx == COND_ONE)
        {
          __sync_fetch_and_add(&n_waiter1_, 1);
          pthread_cond_timedwait(&cond1_, &mutex_, make_abs_end_time(&end_time, timeout_us));
          __sync_fetch_and_add(&n_waiter1_, -1);
        }
        else
        {
          __sync_fetch_and_add(&n_waiter2_, 1);
          pthread_cond_timedwait(&cond2_, &mutex_, make_abs_end_time(&end_time, timeout_us));
          __sync_fetch_and_add(&n_waiter2_, -1);
        }
        return err;
      }

      int signal(const CondIdx idx)
      {
        int err = OB_SUCCESS;
        if (idx == COND_ONE)
        {
          if (n_waiter1_ > 0)
          {
            pthread_cond_signal(&cond1_);
          }
        }
        else
        {
          if (n_waiter2_ > 0)
          {
            pthread_cond_signal(&cond2_);
          }
        }
        return err;
      }
      pthread_mutex_t mutex_;
      pthread_cond_t cond1_;
      pthread_cond_t cond2_;
      volatile int64_t n_waiter1_;
      volatile int64_t n_waiter2_;
      struct Guard
      {
        Guard(ObCondPair* cond, const CondIdx idx, int& err, int64_t timeout_us): cond_(cond), idx_(idx),
                                                                           err_(err), timeout_us_(timeout_us)
        {
          if (NULL != cond_)
          {
            cond_->lock();
          }
        }
        ~Guard()
        {
          if (NULL != cond_)
          {
            if (OB_EAGAIN == err_)
            {
              cond_->wait(idx_, timeout_us_);
            }
            else
            {
              cond_->signal(COND_ONE == idx_? COND_TWO: COND_ONE);
            }
            cond_->unlock();
          }
        }
        ObCondPair* cond_;
        CondIdx idx_;
        int& err_;
        int64_t timeout_us_;
      };
    };

    class Queue
    {
      public:
        const static int64_t WAIT_TIME_US = 10 * 1000;
      public:
        Queue(const bool enable_cond): enable_cond_(enable_cond), capacity_(0), items_(NULL), front_(0), rear_(0) {}
        ~Queue() {}

        int init(void** items, int64_t capacity) {
          int err = OB_SUCCESS;
          items_ = items;
          capacity_ = capacity;
          return err;
        }

        int push(void* p) {
          int err = OB_EAGAIN;
          //ObCondPair::Guard guard(enable_cond_? &cond_: NULL, ObCondPair::COND_ONE, err, WAIT_TIME_US);
          int64_t rear = 0;
          if (NULL == items_)
          {
            err = OB_NOT_INIT;
          }
          while(OB_EAGAIN == err)
          {
            rear = rear_;
            if (front_ + (capacity_<<1) <= rear+1)
            {
              break;
            }
            else if (0 == (rear&1) && front_+(capacity_<<1) > rear+1
                && __sync_bool_compare_and_swap(&rear_, rear, rear+1))
            {
              err = OB_SUCCESS;
            }
          }
          if (OB_SUCCESS == err);
          {
            items_[(rear>>1)%capacity_] = p;
            __sync_fetch_and_add(&rear_, 1);
          }
          return err;
        }

        int pop(void*& p) {
          int err = OB_EAGAIN;
          int64_t front = 0;
          //ObCondPair::Guard guard(enable_cond_? &cond_: NULL, ObCondPair::COND_TWO, err, WAIT_TIME_US);
          if (NULL == items_)
          {
            err = OB_NOT_INIT;
          }
          while(OB_EAGAIN == err)
          {
            front = front_;
            if (front+1 >= rear_)
            {
              break;
            }
            else if (0 == (front&1) && front+1 < rear_
                     && __sync_bool_compare_and_swap(&front_, front, front+1))
            {
              err = OB_SUCCESS;
            }
          }
          if (OB_SUCCESS == err)
          {
            p = items_[(front>>1)%capacity_];
            __sync_fetch_and_add(&front_, 1);
          }
          return err;
        }
      private:
        bool enable_cond_;
        ObCondPair cond_;
        int64_t capacity_;
        void** items_;
        volatile int64_t front_;
        volatile int64_t rear_;
    };

    template<typename T>
    class TypedQueue
    {
      public:
        TypedQueue(const bool enable_cond): buf_(NULL), free_(enable_cond), queue_(enable_cond) {}        
        ~TypedQueue() {
          if (NULL == buf_)
          {
            ob_free(buf_);
            buf_ = NULL;
          }
        }
      public:
        int init(const int64_t capacity) {
          int err = OB_SUCCESS;
          if (0 >= capacity)
          {
            err = OB_INVALID_ARGUMENT;
          }
          else if (NULL != buf_)
          {
            err = OB_INIT_TWICE;
          }
          else if (NULL == (buf_ = (char*)ob_malloc(2 * sizeof(void*) * capacity + sizeof(T) * capacity)))
          {
            err = OB_ALLOCATE_MEMORY_FAILED;
          }
          else if (OB_SUCCESS != (err = free_.init((void**)buf_, capacity)))
          {
            TBSYS_LOG(ERROR, "free_.init(capacity=%ld)=>%d", capacity, err);
          }
          else if (OB_SUCCESS != (err = queue_.init((void**)(buf_ + sizeof(void*) * capacity), capacity)))
          {
            TBSYS_LOG(ERROR, "queue_.init(capacity=%ld)=>%d", capacity, err);
          }
          for(int64_t i = 0; OB_SUCCESS == err && i < capacity; i++)
          {
            err = free_.push(buf_ + 2 * sizeof(void*) * capacity + i * sizeof(T));
          }
          if (OB_SUCCESS != err)
          {
            if (buf_ != NULL)
            {
              ob_free(buf_);
              buf_ = NULL;
            }
          }
          return err;
        }
        int alloc(T*& p) {
          int err = OB_SUCCESS;
          if (OB_SUCCESS != (err = free_.pop((void*&)p)))
          {}
          else if (NULL == (p = new(p)T()))
          {
            err = OB_INIT_FAIL;
          }
          return err;
        }
        int free(T* p) {
          int err = OB_SUCCESS;
          if (OB_SUCCESS != (err = free_.push((void*)p)))
          {}
          else
          {
            p->~T();
          }
          return err;
        }
        int push(T* p) {
          return queue_.push((void*)p);
        }
        int pop(T*& p) {
          return queue_.pop((void*&)p);
        }
      private:
        char* buf_;
        Queue free_;
        Queue queue_;
    };
  }; // end namespace common
}; // end namespace oceanbase

#endif /* __OB_COMMON_OB_ASYNC_QUEUE_H__ */
