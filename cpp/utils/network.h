#ifndef __OB_UTILS_NETWORK_UTILS_H__
#define __OB_UTILS_NETWORK_UTILS_H__
#include "common.h"
#include "func.h"
#include <new>

struct Task;
struct Channel
{
  Lock lock_;
  int64_t ref_;
  int64_t last_access_time_;
  BufHolder buf_holder_;
  Queue queue_;
  Channel(): lock_(), ref_(0), last_access_time_(0), queue_() {}
  ~Channel() {}
  int init(const int64_t queue_len) {
    int err = 0;
    if (0 != (err = queue_.init(queue_len, (void**)buf_holder_.get_buf(queue_len * sizeof(void*)))))
    {
      log(ERROR, "queue_.init(len=%ld)=>%d", queue_len, err);
    }
    return err;
  }

  int alloc() {
    int err = 0;
    bool locked = false;
    if (!lock_.try_lock())
    {
      log(DEBUG, "lock failed.");
      err = EAGAIN;
      locked = true;
    }
    else if (ref_ > 0)
    {
      log(DEBUG, "ref != 0");
      err = EAGAIN;
    }
    else
    {
      ref_++;
    }
    if (locked)
    {
      lock_.unlock();
    }
    return err;
  }

  int ref() {
    int err = 0;
    bool locked = false;
    if (!lock_.try_lock())
    {
      log(DEBUG, "lock failed.");
      err = EAGAIN;
      locked = true;
    }
    else if (ref_ < 0)
    {
      err = EINTERNAL;
    }
    else
    {
      ref_++;
    }
    if (locked)
    {
      lock_.unlock();
    }
    return err;
  }

  int deref() {
    int err = 0;
    bool locked = false;
    if (!lock_.try_lock())
    {
      log(DEBUG, "lock failed.");
      err = EAGAIN;
      locked = true;
    }
    else if (ref_ <= 0)
    {
      err = EINTERNAL;
    }
    else
    {
      ref_--;
    }
    if (locked)
    {
      lock_.unlock();
    }
    return err;
  }

  int send(Task* task) {
    return queue_.push((void*)task);
  }

  int get(Task*& task) {
    return queue_.pop((void*&)task);
  }
};

class Network
{
  private:
    int max_n_channels_;
    Channel* channels_;
    int64_t last_alloc_id_;
  public:
    Network(): max_n_channels_(0), channels_(NULL), last_alloc_id_(0) {}
    ~Network(){
      if (NULL != channels_)
      {
        delete []channels_;
        channels_ = NULL;
      }
    }
    bool is_inited() const {
      return NULL != channels_;
    }
    int init(int64_t n_channels, int64_t queue_len) {
      int err = 0;
      if (is_inited())
      {
        err = EINIT;
      }
      else if (0 > n_channels || 0 > queue_len)
      {
        err = EINVAL;
      }
      else if (NULL == (channels_ = new(std::nothrow)Channel[queue_len]))
      {
        err = ENOMEM;
      }
      for (int64_t i = 0; 0 == err && i < n_channels; i++)
      {
        if (0 != (err = channels_[i].init(queue_len)))
        {
          log(ERROR, "channels[%ld].init(queue_len=%ld)=>%d", i, queue_len, err);
        }
      }
      return err;
    }

    int alloc(int64_t& id) {
      int err = 0;
      int64_t try_id = last_alloc_id_;
      int64_t i = 0;
      if (!is_inited())
      {
        err = EINIT;
      }
      for(i = 0; 0 == err && i < max_n_channels_; i++)
      {
        if (0 == (err = channels_[try_id].alloc()))
        {
          break;
        }
        else if (0 != EAGAIN)
        {
          log(ERROR, "channels[%ld].alloc()=>%d", try_id, err);

        }
        else
        {
          try_id = (try_id+1)% max_n_channels_;
        }
      }

      if (0 == err)
      {
        id = try_id;
        last_alloc_id_ = try_id;
      }
      return err;
    }

    int free(const int64_t id) {
      int err = 0;
      if (!is_inited())
      {
        err = EINIT;
      }
      else if (0 > id && id <= max_n_channels_)
      {
        err = EINVAL;
      }
      else if (0 != (err = channels_[id].deref()))
      {
        log(ERROR, "channels_[%ld].free()=>%d", id, err);
      }
      return err;
    }

    int send(const int64_t id, Task* task) {
      int err = 0;
      if (!is_inited())
      {
        err = EINIT;
      }
      else if (0 > id && id <= max_n_channels_)
      {
        err = EINVAL;
      }
      else if (0 != (err = channels_[id].send(task)))
      {
        if (EAGAIN != err)
        {
          log(ERROR, "channels_[%ld].send(task[%p])=>%d", id, task, err);
        }
        else
        {
          log(DEBUG, "channels_[%ld].send(task[%p])=>%d", id, task, err);
        }
      }
      return err;
    }

    int get(const int64_t id, Task*& task) {
      int err = 0;
      if (!is_inited())
      {
        err = EINIT;
      }
      else if (0 > id && id <= max_n_channels_)
      {
        err = EINVAL;
      }
      else if (0 != (err = channels_[id].get(task)))
      {
        if (EAGAIN != err)
        {
          log(ERROR, "channels_[%ld].get(task)=>%d", id, err);
        }
      }
      return err;
    }
};
#endif /* __OB_UTILS_NETWORK_UTILS_H__ */
