#ifndef __OB_UTILS_THREAD_UTILS_H__
#define __OB_UTILS_THREAD_UTILS_H__

#include <pthread.h>
#include "common.h"

class ThreadWorker;
typedef void* (*pthread_handler_t)(void*);
class IRunnable
{
  public:
    IRunnable() {}
    virtual ~IRunnable(){}
    virtual int do_work(ThreadWorker* worker) = 0;
};
struct CallRunnableArgs
{
  const char* name_;
  IRunnable* runnable_;
  ThreadWorker* thread_worker_;
};

inline int call_runnable(CallRunnableArgs* arg)
{
  int err = 0;
  if (NULL == arg)
  {
    err = EINVAL;
  }
  else
  {
    log(INFO, "#%s START.", arg->name_);
    if (0 != (err = arg->runnable_->do_work(arg->thread_worker_)))
    {
      log(ERROR, "#%s FINISHED with err=%d", arg->name_, err);
    }
    else
    {
      log(INFO, "#%s FINISHED success.", arg->name_);
    }
  }
  return err;
}

class ThreadWorker
{
  public:
    volatile bool stop_;
    int64_t self_addr_;
    const char* name_;
  private:    
    pthread_t thread_;
    IRunnable* runnable_;
    CallRunnableArgs arg;
  public:
    ThreadWorker(): stop_(false), runnable_(NULL) {}
    ~ThreadWorker() {}
    bool is_inited() const {
      return NULL != name_ && NULL != runnable_;
    }
    int init(const char* name, IRunnable* runnable, int64_t self_addr) {
      int err = 0;
      if (is_inited())
      {
        err = EINIT;
      }
      else if (NULL == runnable || 0 > self_addr)
      {
        err = EINVAL;
      }
      else
      {
        self_addr_ = self_addr;
        runnable_ = runnable;
        name_ = name;
      }
      return err;
    }

    int start() {
      arg.name_ = name_;
      arg.runnable_ = runnable_;
      arg.thread_worker_ = this;
      return pthread_create(&thread_, NULL, (pthread_handler_t)call_runnable, &arg) == 0? 0: EINTERNAL;
    }
    int stop() {
      log(INFO, "#%s Request STOP.", name_);
      stop_ = true;
      return 0;
    }
    int wait() {
      int64_t thread_err = 0;
      int err = pthread_join(thread_, (void**)&thread_err) == 0? 0: EINTERNAL;
      return err;
    }
    int64_t get_addr() const {
      return self_addr_;
    }
};

#endif /* __OB_UTILS_THREAD_UTILS_H__ */
