#ifndef __OB_AX_ECHO_SERVER_H__
#define __OB_AX_ECHO_SERVER_H__
#include "io_sched.h"

struct EchoServerSock: public Sock
{
  EchoServerSock(): Sock(NORMAL), last_active_time_(get_us()) {}
  virtual ~EchoServerSock() {}
  int clone(Sock*& other) {
    int err = AX_SUCCESS;
    other = new EchoServerSock();
    return err;
  }
  int destroy() {
    int err = AX_SUCCESS;
    Sock::destroy();
    delete this;
    return err;
  }
  bool kill() {
    int64_t idle_timeout = 5 * 1000000;
    return last_active_time_ + idle_timeout < get_us();
  }
  void mark_active() { last_active_time_ = get_us(); }
  int read(IOSched* sched) {
    int err = AX_SUCCESS;;
    char buf[64];
    ssize_t rbytes = 0;
    UNUSED(sched);
    mark_active();
    if ((rbytes = ::read(fd_, buf, sizeof(buf))) < 0)
    {
      if (errno == EINTR)
      {}
      else if (EAGAIN == errno || EWOULDBLOCK == errno)
      {
        err = AX_EAGAIN;
      }
    }
    else if (rbytes == 0)
    {
      err = AX_SOCK_HUP;
    }
    else
    {
      ::write(fd_, buf, rbytes);
    }
    MLOG(INFO, "fd=%d read: %.*s", fd_, rbytes, buf);
    return err;
  }
  int write(IOSched* sched) {
    int err = AX_EAGAIN;
    mark_active();
    MLOG(INFO, "fd=%d become writable, ignore", fd_);
    return err;
  }
  int64_t last_active_time_;
};

class EchoServer
{
public:
  typedef EchoServerSock Sock;
  EchoServer(): stop_(false), thread_num_(0) {}
  ~EchoServer(){}
public:
  int init(int port, int thread_num) {
    int err = AX_SUCCESS;
    struct sockaddr_in addr;
    const char* ip = "127.0.0.1";
    int64_t capacity = 1<<16;
    if (AX_SUCCESS != (err = io_sched_.init(capacity, (char*)ax_malloc(io_sched_.calc_mem_usage(capacity)))))
    {
      MLOG(ERROR, "io_sched.init()=>%d", err);
    }
    else if (AX_SUCCESS != (err = epoll_sock_.init(&io_sched_)))
    {
      MLOG(ERROR, "epoll_sock.init()=>%d", err);
    }
    else if (AX_SUCCESS != (err = listen_sock_.init(epoll_sock_.fd_, &sock_, make_sockaddr(&addr, inet_addr(ip), port), &io_sched_)))
    {
      MLOG(ERROR, "listen_sock.init()=>%d", err);
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
      io_sched_.sched(100 * 1000);
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
  IOSched io_sched_;
  EpollSock epoll_sock_;
  ListenSock listen_sock_;
  Sock sock_;
};

#endif /* __OB_AX_ECHO_SERVER_H__ */
