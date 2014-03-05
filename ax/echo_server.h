#ifndef __OB_AX_ECHO_SERVER_H__
#define __OB_AX_ECHO_SERVER_H__
#include "nio.h"

struct EchoSockHandler: public SockHandler
{
  EchoSockHandler() {}
  virtual ~EchoSockHandler() {}
  int clone(SockHandler*& other) {
    int err = AX_SUCCESS;
    other = new EchoSockHandler();
    return err;
  }
  int destroy() {
    int err = AX_SUCCESS;
    before_destroy();
    if (fd_ > 0)
    {
      close(fd_);
    }
    delete this;
    return err;
  }
  int read() {
    int err = AX_SUCCESS;;
    MLOG(INFO, "read");
    return err;
  }
  int write() {
    int err = AX_SUCCESS;;
    MLOG(INFO, "read");
    return err;
  }
};

class EchoServer
{
public:
  typedef EchoSockHandler Sock;
    EchoServer(): stop_(false), thread_num_(0) {}
  ~EchoServer(){}
public:
  int init(int port, int thread_num) {
    int err = AX_SUCCESS;
    int64_t capacity = 1<<16;
    Server server;
    Sock* sock = new Sock();
    server.ip_ = inet_addr("127.0.0.1");
    server.port_ = port;
    if (AX_SUCCESS != (err = nio_.init(sock, capacity, (char*)ax_malloc(nio_.calc_mem_usage(capacity)))))
    {
      MLOG(ERROR, "nio.init()=>%d", err);
    }
    else if (AX_SUCCESS != (err = nio_.listen(server)))
    {
      MLOG(ERROR, "listen fail on port: port=%d", port);
    }
    else
    {
      thread_num_ = thread_num;
      MLOG(INFO, "echo_server init: port=%d thread_num=%ld", port, thread_num);
    }
    return err;
  }
  int mainloop() {
    int err = AX_SUCCESS;
    while(!stop_) {
      nio_.do_loop(100 * 1000);
    }
    return err;
  }
  int start() {
    int sys_err = 0;
    int err = AX_SUCCESS;
    pthread_t thread[32];
    for(int64_t i = 0; i < min(thread_num_, (int64_t)arrlen(thread)); i++) {
      if (0 != (sys_err = pthread_create(thread + i, NULL, (void* (*)(void*))thread_func, (void*)this)))
      {
        err = AX_FATAL_ERR;
        MLOG(ERROR, "pthread_create fail, err=%d", sys_err);
      }
    }
    for(int64_t i = 0; i < min(thread_num_, (int64_t)arrlen(thread)); i++) {
      pthread_join(thread[i], NULL);
    }
    return err;
  }
  static int thread_func(EchoServer* self) {
    return self->mainloop();
  }
private:
  bool stop_;
  int64_t thread_num_;
  Nio nio_;
};

#endif /* __OB_AX_ECHO_SERVER_H__ */
