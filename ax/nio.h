#ifndef __OB_AX_NIO_H__
#define __OB_AX_NIO_H__
#include "io_sched.h"


enum {
  SOCK_FLAG_EPOLL = 1,
  SOCK_FLAG_REARM = 2,
  SOCK_FLAG_LISTEN = 4,
  SOCK_FLAG_CONNECTING = 8,
  SOCK_FLAG_NORMAL = 16
};

struct EpollSock
{
  EpollSock(): id_(INVALID_ID), flag_(), fd_(-1), epfd_(-1) {
    event_.data.u64 = 0;
    event_.events = 0;
  }
  virtual ~EpollSock() {}
  virtual int clone(SockHandler*& other) = 0;
  virtual int destroy() = 0;
  virtual int read() = 0;
  virtual int write() = 0;
  SockHandler* set_args(Id id, int flag, int fd, int epfd, uint32_t events_flag) {
    id_ = id;
    flag_ = flag;
    fd_ = fd;
    epfd_ = epfd;
    event_.data.u64 = id;
    event_.events = events_flag;
    return this;
  }
  bool try_lock() { return lock_.try_lock(); }
  void unlock() { lock_.unlock(); }
  uint32_t get_event_flag() { return event_.events; }
  void after_create() {
    MLOG(INFO, "create_sock: id=%lx fd=%d type=%d", id_, fd_, flag_);
  }
  void before_destroy() {
    MLOG(INFO, "destroy_sock: id=%lx fd=%d type=%d", id_, fd_, flag_);
  }
  SpinLock lock_;
  Id id_;
  int flag_;
  int fd_;
  int epfd_;
  epoll_event event_;
};

class Nio
{
public:
  typedef struct epoll_event Event;
  Nio(): evfd_(-1), epfd_(-1), handler_(NULL) {}
  ~Nio() { destroy(); }
public:
  int init(SockHandler* handler, int64_t capacity, char* buf) {
    int err = AX_SUCCESS;
    MemChunkCutter cutter(calc_mem_usage(capacity), buf);
    if (capacity <= 0 || !is2n(capacity) || NULL == handler || NULL == buf)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL != handler_)
    {
      err = AX_INIT_TWICE;
    }
    else if (AX_SUCCESS != (err = init_container(ev_free_list_, capacity, cutter)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = init_container(ev_queue_, capacity, cutter)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = init_container(out_sock_table_, capacity, cutter)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = init_container(sock_map_, capacity, cutter)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = init_container(rearm_queue_, capacity, cutter)))
    {
      ERR(err);
    }
    else
    {
      handler_ = handler;
    }
    for(int64_t i = 0; AX_SUCCESS == err && i < capacity; i++)
    {
      err = ev_free_list_.push(cutter.alloc(sizeof(Event)));
    }
    if (AX_SUCCESS != err)
    {}
    else if (AX_SUCCESS != (err = create_epoll_fd()))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = create_event_fd()))
    {
      ERR(err);
    }
    if (AX_SUCCESS != err)
    {
      destroy();
    }
    return err;
  }
  int64_t calc_mem_usage(int64_t capacity) {
    return ev_free_list_.calc_mem_usage(capacity) + ev_queue_.calc_mem_usage(capacity) + out_sock_table_.calc_mem_usage(capacity)
      + sock_map_.calc_mem_usage(capacity) + rearm_queue_.calc_mem_usage(capacity)
      + sizeof(Event) * capacity;
  }
  void destroy() {
    ev_free_list_.destroy();
    ev_queue_.destroy();
    out_sock_table_.destroy();
    sock_map_.destroy();
    if (evfd_ >= 0)
    {
      close(evfd_);
      evfd_ = -1;
    }
    if (epfd_ >= 0)
    {
      close(epfd_);
      epfd_ = -1;
    }
    handler_ = NULL;
  }
  int listen(Server& server) {
    int err = AX_SUCCESS;
    int backlog = 128;
    Id id = INVALID_ID;
    struct sockaddr_in sin;
    int fd = -1;
    if (!server.is_valid())
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (epfd_ < 0)
    {
      err = AX_NOT_INIT;
    }
    else if ((fd = ::socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
      err = AX_SOCK_CREATE_ERR;
      ERR(err);
    }
    else if (0 != ::bind(fd, (sockaddr*)make_sockaddr(&sin, server.ip_, server.port_), sizeof(sin)))
    {
      err = AX_SOCK_BIND_ERR;
      ERR(err);
    }
    else if (0 != ::listen(fd, backlog))
    {
      err = AX_SOCK_LISTEN_ERR;
    }
    else if (0 != make_fd_nonblocking(fd))
    {
      err = AX_FCNTL_ERR;
    }
    else if (AX_SUCCESS != (err = alloc_sock(id, fd, SOCK_FLAG_LISTEN, EPOLLIN | EPOLLONESHOT)))
    {
      ERR(err);
    }
    if (AX_SUCCESS != err)
    {
      if (fd >= 0)
      {
        close(fd);
      }
    }
    return err;
  }
protected:
  int handle_listen_sock(SockHandler* sock, Event* event) {
    int err = AX_SUCCESS;
    int fd = -1;
    Id id = INVALID_ID;
    if ((fd = accept(sock->fd_, NULL, NULL)) < 0)
    {
      if (errno != EAGAIN && errno != EWOULDBLOCK)
      {
        err = AX_SOCK_ACCEPT_ERR;
      }
    }
    else if (AX_SUCCESS != (err = (alloc_sock(id, fd, SOCK_FLAG_NORMAL, EPOLLIN | EPOLLONESHOT))))
    {}
    if (AX_SUCCESS != err)
    {
      if (fd >= 0)
      {
        close(fd);
      }
    }
    return err;
  }
  int create_sock_handler(SockHandler*& handler) {
    return NULL == handler_? AX_NOT_INIT: handler_->clone(handler);
  }
  int destroy_sock_handler(SockHandler* handler) {
    return NULL == handler? AX_INVALID_ARGUMENT: handler->destroy();
  }

  int alloc_sock(Id& id, int fd, int flag, uint32_t events_flag) {
    int err = AX_SUCCESS;
    epoll_event event;
    SockHandler* sock = NULL;
    if (AX_SUCCESS != (err = create_sock_handler(sock)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = sock_map_.add(id, sock->set_args(INVALID_ID, flag, fd, epfd_, events_flag))))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = sock_map_.fetch(id, (void*&)sock)))
    {
      ERR(err);
    }
    else
    {
      sock->id_ = id;
      sock->after_create();
      sock_map_.revert(id);
      if (flag != SOCK_FLAG_EPOLL && 0 != epoll_ctl(epfd_, EPOLL_CTL_ADD, fd, make_epoll_event(&event, events_flag, id)))
      {
        err = AX_EPOLL_CTL_ERR;
      }
    }
    if (AX_SUCCESS != err)
    {
      if (id != INVALID_ID)
      {
        free_sock(id);
      }
      else if (NULL != sock)
      {
        destroy_sock_handler(sock);
      }
    }
    return err;
  }
  int free_sock(Id id) {
    int err = AX_SUCCESS;
    SockHandler* sock = NULL;
    if (AX_SUCCESS != (err = sock_map_.del(id, (void*&)sock)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = destroy_sock_handler(sock)))
    {
      ERR(err);
    }
    return err;
  }
  int create_epoll_fd() {
private:
  IOSchedule io_schedule_;
  EpollSock epoll_sock_;
  ListenSock listen_sock_;
  Sock* sock_;
};

#endif /* __OB_AX_NIO_H__ */
