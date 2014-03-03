#ifndef __OB_AX_ECHO_SERVER_H__
#define __OB_AX_ECHO_SERVER_H__
#include "nio.h"

struct EchoSockHandler: public SockHandler
{
  EchoSockHandler() {}
  virtual ~EchoSockHandler() {}
  int clone(SockHandler*& other) {
    other = new EchoSockHandler();
    return err;
  }
  int destroy() {
    delete this;
  }
  int read() {
    int err;
    MLOG(INFO, "read");
    return err;
  }
  int write() {
    int err;
    MLOG(INFO, "read");
    return err;
  }
};

class EchoServer
{
public:
  typedef EchoSockHandler Sock;
  EchoServer(){}
  ~EchoServer(){}
public:
  int init(int port, int thread_num) {
    int err = AX_SUCCESS;
    int64_t capacity = 1<<16;
    Sock* sock = new Sock();
    if (AX_SUCCESS != (err = nio_.init(sock, capacity, ax_malloc(nio_.calc_mem_usage(capacity)))))
    {
      MLOG(ERROR, "nio.init()=>%d", err);
    }
    return err;
  }
  int mainloop() {
    while(!stop_) {
      nio_.do_loop();
    }
  }
  int start() {
    int sys_err = 0;
    int err = AX_SUCCESS;
    pthread_t thread[32];
    for(int64_t i = 0; i < arrlen(thread); i++) {
      if (0 != (sys_err = pthread_create(thread + i, NULL, thread_func, this)))
      {
        err = AX_FATAL_ERR;
        MLOG(ERROR, "pthread_create fail, err=%d", sys_err);
      }
    }
    for(int64_t i = 0; i < arrlen(thread); i++) {
      pthread_join(thread[i], NULL);
    }
    return err;
  }
  static int thread_func(EchoServer* this) {
    return this->mainloop();
  }
private:
  Nio nio_;
};

#endif /* __OB_AX_ECHO_SERVER_H__ */
