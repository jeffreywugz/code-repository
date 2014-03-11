#ifndef __OB_AX_ECHO_SERVER_H__
#define __OB_AX_ECHO_SERVER_H__
#include "nio.h"

class EchoHandler: public PacketHandler
{
public:
  EchoHandler() {}
  virtual ~EchoHandler() {}
public:
  int init(Nio* nio, int64_t max_pkt_num) {
    int err = AX_SUCCESS;
    UNUSED(max_pkt_num);
    nio_ = nio;
    return err;
  }
  int alloc_packet(Packet*& pkt) {
    int err = AX_SUCCESS;
    pkt = new Packet();
    return err;
  }
  int free_packet(Packet* pkt) {
    int err = AX_SUCCESS;
    if (NULL != pkt)
    {
      delete pkt;
    }
    return err;
  }
  int handle_packet(Id id, Packet* pkt) {
    int err = AX_SUCCESS;
    if (INVALID_ID == id || NULL == pkt)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL == nio_)
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS != (err = nio_->send_packet(id, pkt)))
    {
      MLOG(WARN, "send_packet()=>%d", err);
    }
    if (NULL != pkt)
    {
      MLOG(INFO, "receive pkt: len=%d content=%s", pkt->len_, pkt->payload_);
    }
    return err;
  }
private:
  Nio* nio_;
};

class EchoServer
{
public:
  EchoServer(): stop_(false), thread_num_(0) {}
  ~EchoServer(){}
public:
  int init(int port, int thread_num) {
    int err = AX_SUCCESS;
    int64_t max_pkt_num = 1<<10;
    int64_t capacity = 1<<16;
    if (AX_SUCCESS != (err = handler_.init(&nio_, max_pkt_num)))
    {
      MLOG(ERROR, "handler.init()=>%d", err);
    }
    else if (AX_SUCCESS != (err = nio_.init(&handler_, port, capacity, (char*)ax_malloc(nio_.calc_mem_usage(capacity)))))
    {
      MLOG(ERROR, "io_sched.init()=>%d", err);
    }
    else
    {
      thread_num_ = thread_num;
      MLOG(INFO, "echo_server init: port=%d thread_num=%d", port, thread_num);
    }
    return err;
  }
  int mainloop() {
    int err = AX_SUCCESS;
    while(!stop_) {
      nio_.sched(100 * 1000);
    }
    return err;
  }
  int start() {
    int sys_err = 0;
    int err = AX_SUCCESS;
    for(int64_t i = 0; i < min(thread_num_, (int64_t)arrlen(thread_)); i++) {
      if (0 != (sys_err = pthread_create(thread_ + i, NULL, (void* (*)(void*))thread_func, (void*)this)))
      {
        err = AX_FATAL_ERR;
        MLOG(ERROR, "pthread_create fail, err=%d", sys_err);
      }
    }
    return err;
  }
  int stop() {
    stop_ = true;
    return AX_SUCCESS;
  }
  int wait() {
    for(int64_t i = 0; i < min(thread_num_, (int64_t)arrlen(thread_)); i++) {
      pthread_join(thread_[i], NULL);
    }
    return AX_SUCCESS;
  }
  static int thread_func(EchoServer* self) {
    return self->mainloop();
  }
  Nio& get_nio() { return nio_; }
private:
  bool stop_;
  int64_t thread_num_;
  pthread_t thread_[32];
  Nio nio_;
  EchoHandler handler_;
};

#endif /* __OB_AX_ECHO_SERVER_H__ */
