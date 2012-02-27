#ifndef __UTILS_SIMU_UTILS_H__
#define __UTILS_SIMU_UTILS_H__
#include "common.h"
#include "func.h"
#include "network.h"
#include "thread.h"

struct Task: public Lock
{
  Task(): ref_(0), code_(0), value_(0), finished_(0), err_(0) {}
  ~Task() {}
  int64_t ref_;
  int64_t code_;
  int64_t value_;
  bool finished_;
  int err_;
  bool finished()
  {
    return finished_;
  }
  void done(int err)
  {
    err_ = err;
    finished_ = true;
  }
};

class TaskAllocator
{
  private:
    Stack free_tasks_;
    int64_t n_tasks_;    
    BufHolder buf_holder_;
  public:
    TaskAllocator() : n_tasks_(0) {}
    ~TaskAllocator() {}
    bool is_inited() const {
      return n_tasks_ > 0;
    }
    int init(int64_t n_tasks){
      int err = 0;
      Task* tasks = NULL;
      if (is_inited())
      {
        err = EINIT;
      }
      else if(0 >= n_tasks)
      {
        err = EINVAL;
      }
      else if (0 != (err = free_tasks_.init(n_tasks, (void**)buf_holder_.get_buf(n_tasks * sizeof(Task*)))))
      {
        log(ERROR, "free_tasks.init(n_tasks=%ld)=>%d", n_tasks, err);
      }
      else if (NULL == (tasks = (Task*)buf_holder_.get_buf(n_tasks * sizeof(Task))))
      {
        err = ENOMEM;
      }
      for (int64_t i = 0; 0 == err && i < n_tasks; i++)
      {
        if (0 != (err = free_tasks_.push(tasks + i)))
        {
          log(ERROR, "free_tasks.push(i=%ld)=>%d", i, err);
        }
      }
      if (0 == err)
      {
        n_tasks_ = n_tasks;
      }
      return err;
    }
    int alloc(Task*& task) {
      return free_tasks_.pop((void*&)task);
    }
    int free(Task* task) {
      return free_tasks_.push((void*)task);
    }
};

class Client
{
  private:
    TaskAllocator* allocator_;
    Network* network_;
  public:
    Client(): allocator_(NULL), network_(NULL) {}
    ~Client() {}
    bool is_inited() const {
      return NULL != allocator_ && NULL != network_;
    }

    int init(TaskAllocator* allocator, Network* network) {
      int err = 0;
      if (is_inited())
      {
        err = EINIT;
      }
      else if (NULL == allocator || NULL == network)
      {
        err = EINVAL;
      }
      else
      {
        allocator_ = allocator;
        network_ = network;
      }
      return err;
    }

    int send_request(const int64_t addr, int64_t code, int64_t val, int64_t timeout_us=1000*1000) {
      int err = 0;
      int wait_err = ETIMEDOUT;
      Task* task = NULL;
      int64_t end_time = get_usec() + timeout_us;
      if (0 != (err = post_request(addr, code, val, task)))
      {
        log(ERROR, "post_request()=>%d", err);
      }
      while(0 == err && get_usec() < end_time)
      {
        if (task->finished())
        {
          wait_err = 0;
          break;
        }
      }
      if (0 == err)
      {
        allocator_->free(task);
      }
      return 0 == err? wait_err: err;
    }

    int post_request(const int64_t addr, int64_t code, int64_t val, Task*& task) {
      int err = 0;
      bool allocated = false;
      if (0 != (err = allocator_->alloc(task)))
      {
        log(ERROR, "allocat_task()=>%d", err);
      }
      else
      {
        allocated = true;
        task->value_ = val;
      }

      if (0 != err)
      {}
      else if (0 != (err = network_->send(addr, task)))
      {
        if (EAGAIN != err)
        {
          log(ERROR, "send_request(code=%ld, value=%ld)=>%d", code, val, err);
        }
        else
        {
          log(DEBUG, "send_request(code=%ld, value=%ld)=>%d", code, val, err);
        }
      }

      if (allocated && 0 != err)
      {
        allocator_->free(task);
      }
      return err;
    }
};

struct AsyncRequest
{
  int64_t request_sent_;
  int64_t request_received_;
  int64_t input_value_;
  int64_t output_value_;
};

int init_thread_worker(const char* name, ThreadWorker& thread_worker, IRunnable* runnable, Network* network)
{
  int err = 0;
  int64_t self_addr = 0;
  if (NULL == runnable || NULL == network)
  {
    err = EINVAL;
  }
  else if (0 != (err = network->alloc(self_addr)))
  {
    log(ERROR, "network.alloc()=>%d", err);
  }
  else if (0 != (err = thread_worker.init(name, runnable, self_addr)))
  {
    log(ERROR, "thread_worker.init(runnable, network)=>%d", err);
  }
  return err;
}

#endif /* __UTILS_SIMU_UTILS_H__ */
