#ifndef __OB_AX_THREAD_WORKER_H__
#define __OB_AX_THREAD_WORKER_H__
#include "common.h"

class IThreadCallable
{
public:
  IThreadCallable() {}
  virtual ~IThreadCallable() {}
  virtual int do_thread_work() = 0;
  static int thread_func(IThreadCallable* self) {
    return self->do_thread_work();
  }
};

class ThreadWorker
{
public:
  ThreadWorker(): thread_num_(0) {
    memset(thread_, 0, sizeof(thread_));
  }
  ~ThreadWorker() { wait(); }
public:
  int start(IThreadCallable* callable, int thread_num=1) {
    int sys_err = 0;
    int err = AX_SUCCESS;
    if (NULL == callable || thread_num <= 0)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (thread_num_ > 0)
    {
      err = AX_INIT_TWICE;
    }
    else
    {
      thread_num_ = thread_num;
    }
    for(int64_t i = 0; AX_SUCCESS == err && i < min(thread_num, (int64_t)arrlen(thread_)); i++) {
      if (0 != (sys_err = pthread_create(thread_ + i, NULL, (void* (*)(void*))IThreadCallable::thread_func, (void*)callable)))
      {
        err = AX_FATAL_ERR;
        MLOG(ERROR, "pthread_create fail, err=%d", sys_err);
      }
    }
    return err;
  }
  void wait() {
    if (thread_num_ > 0)
    {
      for(int64_t i = 0; i < min(thread_num_, (int64_t)arrlen(thread_)); i++) {
        pthread_join(thread_[i], NULL);
      }
      thread_num_ = 0;
    }
  }
private:
  int thread_num_;
  pthread_t thread_[64];
};
#endif /* __OB_AX_THREAD_WORKER_H__ */
