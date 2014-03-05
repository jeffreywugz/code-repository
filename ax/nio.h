#ifndef __OB_AX_NIO_H__
#define __OB_AX_NIO_H__
#include "common.h"
#include <sys/epoll.h>
#include <sys/eventfd.h>

enum {
  SOCK_FLAG_EPOLL = 1,
  SOCK_FLAG_REARM = 2,
  SOCK_FLAG_LISTEN = 4,
  SOCK_FLAG_CONNECTING = 8,
  SOCK_FLAG_NORMAL = 16
};

struct SockHandler
{
  SockHandler(): id_(INVALID_ID), flag_(), fd_(-1), epfd_(-1) {
    event_.data.u64 = 0;
    event_.events = 0;
  }
  virtual ~SockHandler() {}
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
  SpinLock lock_;
  Id id_;
  int flag_;
  int fd_;
  int epfd_;
  epoll_event event_;
};

inline int make_fd_nonblocking(int fd)
{
  int err = 0;
  int flags = 0;
  if ((flags = fcntl(fd, F_GETFL, 0)) < 0 || fcntl(fd, F_SETFL, flags|O_NONBLOCK) < 0)
  {
    err = -errno;
  }
  return err;
}

inline struct sockaddr_in* make_sockaddr(struct sockaddr_in* sin, in_addr_t ip, int port)
{
  if (NULL != sin)
  {
    sin->sin_port = htons(port);
    sin->sin_addr.s_addr = ip;
    sin->sin_family = AF_INET;
  }
  return sin;
}

inline struct epoll_event* make_epoll_event(epoll_event* event, uint32_t event_flag, Id id)
{
  if (NULL != event)
  {
    event->events = event_flag;
    event->data.u64 = id;
  }
  return event;
}

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
  int do_loop(int64_t timeout_us) {
    int err = AX_SUCCESS;
    Event* event = NULL;
    SockHandler* sock = NULL;
    Id id = INVALID_ID;
    if (AX_SUCCESS != (err = ev_queue_.pop((void*&)event, timeout_us))
        && AX_EAGAIN != err)
    {
      ERR(err);
    }
    else if (AX_SUCCESS != err)
    {}
    else if (AX_SUCCESS != (err = sock_map_.fetch((id = event->data.u64), (void*&)sock)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = handle_event(sock, event))
             && AX_NIO_CAN_NOT_LOCK != err)
    {
      sock_map_.revert(id);
      free_sock(id);
    }
    else if (AX_NIO_CAN_NOT_LOCK == err)
    {}
    else if (sock->flag_ != SOCK_FLAG_EPOLL)
    {
      Event event;
      if (0 != epoll_ctl(sock->epfd_, EPOLL_CTL_MOD, sock->fd_, make_epoll_event(&event, id, sock->get_event_flag())))
      {
        err = AX_EPOLL_CTL_ERR;
      }
      sock_map_.revert(id);
    }
    if (NULL != event)
    {
      int tmperr = AX_SUCCESS;
      if (AX_SUCCESS != (tmperr = ev_free_list_.push(event)))
      {
        err = tmperr;
      }
    }
    return err;
  }
protected:
  int handle_rearm_queue() {
    int err = AX_SUCCESS;
    int tmperr = AX_SUCCESS;
    Id id = INVALID_ID;
    SockHandler* sock = NULL;
    while(AX_SUCCESS == err)
    {
      if (AX_SUCCESS != (err = rearm_queue_.pop((void*&)id)))
      {
        ERR(err);
      }
      else if (AX_SUCCESS != (tmperr = sock_map_.fetch(id, (void*&)sock)))
      {
        ERR(err);
      }
      else
      {
        Event event;
        if (0 != epoll_ctl(sock->epfd_, EPOLL_CTL_MOD, sock->fd_, make_epoll_event(&event, id, sock->get_event_flag())))
        {
          tmperr = AX_EPOLL_CTL_ERR;
        }
        sock_map_.revert(id);
        if (AX_SUCCESS != tmperr)
        {
          free_sock(id);
        }
      }
    }
    return err;
  }
  int handle_epoll_fd(SockHandler* sock, Event* event) {
    int err = AX_SUCCESS;
    Event events[32];
    int count = 0;
    int timeout = 100 * 1000;
    if (AX_SUCCESS != (err = handle_rearm_queue()))
    {
      ERR(err);
    }
    else if ((count = epoll_wait(sock->fd_, events, arrlen(events) - 1, timeout)) < 0)
    {
      err = AX_EPOLL_WAIT_ERR;
    }
    else
    {
      events[count++] = *event;
      for(int i = 0; i < count; i++)
      {
        epoll_event* event = NULL;
        while(AX_SUCCESS != ev_free_list_.pop((void*&)event))
          ;
        *event = events[i];
        while(AX_SUCCESS != ev_queue_.push(event))
          ;
      }
    }
    return err;
  }
  int handle_rearm_fd(SockHandler* sock, Event* event) {
    int err = AX_SUCCESS;
    eventfd_t value = 0;
    if (0 != eventfd_read(sock->fd_, &value))
    {
      err = AX_EVENTFD_CREATE_ERR;
    }
    return err;
  }
  int handle_listen_sock(SockHandler* sock, Event* event) {
    int err = AX_SUCCESS;
    int fd = -1;
    Id id = INVALID_ID;
    if ((fd= accept(sock->fd_, NULL, NULL)) < 0)
    {
      if (errno != EAGAIN && errno != EWOULDBLOCK)
      {
        err = AX_ACCEPT_ERR;
      }
    }
    else if (AX_SUCCESS != (err = (alloc_sock(id, fd, 0, EPOLLIN | EPOLLONESHOT))))
    {}
    return err;
  }
  int handle_normal_sock(SockHandler* sock, Event* event) {
    int err = AX_SUCCESS;
    if ((event->events & EPOLLOUT)
        && (AX_SUCCESS != (err = sock->write())))
    {}
    else if ((event->events & EPOLLIN)
             && (AX_SUCCESS != (err = sock->read())))
    {}
    return err;
  }
  int handle_connecting_sock(SockHandler* sock, Event* event) {
    int err = AX_SUCCESS;
    int sys_err = 0;
    socklen_t errlen = sizeof(sys_err);
    if (0 != getsockopt(sock->fd_, SOL_SOCKET, SO_ERROR, &sys_err, &errlen))
    {
      err = AX_GET_SOCKOPT_ERR;
      close(sock->fd_);
    }
    else
    {
      sock->flag_ = SOCK_FLAG_NORMAL;
    }
    return err;
  }
  int handle_event(SockHandler* sock, epoll_event* event) {
    int err = AX_SUCCESS;
    bool locked = false;
    if (NULL == sock)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (!sock->try_lock())
    {
      locked = true;
    }
    else
    {
      switch(sock->flag_)
      {
        case SOCK_FLAG_NORMAL:
          err = handle_normal_sock(sock, event);
          break;
        case SOCK_FLAG_EPOLL:
          err = handle_epoll_fd(sock, event);
          break;
        case SOCK_FLAG_REARM:
          err = handle_rearm_fd(sock, event);
          break;
        case SOCK_FLAG_LISTEN:
          err = handle_listen_sock(sock, event);
          break;
        case SOCK_FLAG_CONNECTING:
          err = handle_connecting_sock(sock, event);
          break;
        default:
          err = AX_NOT_SUPPORT;
      };
    }
    if (locked)
    {
      sock->unlock();
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
    SockHandler* handler = NULL;
    if (AX_SUCCESS != (err = create_sock_handler(handler)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = sock_map_.add(id, handler->set_args(INVALID_ID, flag, fd, epfd_, events_flag))))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = sock_map_.fetch(id, (void*&)handler)))
    {
      ERR(err);
    }
    else
    {
      handler->id_ = id;
      sock_map_.revert(id);
      if (flag != SOCK_FLAG_EPOLL && 0 != epoll_ctl(epfd_, EPOLL_CTL_ADD, fd, make_epoll_event(&event, id, events_flag)))
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
      else if (NULL != handler)
      {
        destroy_sock_handler(handler);
      }
    }
    return err;
  }
  int free_sock(Id id) {
    int err = AX_SUCCESS;
    SockHandler* handler = NULL;
    if (AX_SUCCESS != (err = sock_map_.del(id, (void*&)handler)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = destroy_sock_handler(handler)))
    {
      ERR(err);
    }
    return err;
  }
  int create_epoll_fd() {
    int err = AX_SUCCESS;
    Id id = INVALID_ID;
    int epfd = -1;
    struct epoll_event* event = NULL;
    if ((epfd = epoll_create1(EPOLL_CLOEXEC)) < 0)
    {
      err = AX_EPOLL_CREATE_ERR;
    }
    else if (AX_SUCCESS != (err = alloc_sock(id, epfd, SOCK_FLAG_EPOLL, 0)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = ev_free_list_.pop((void*&)event)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = ev_queue_.push(event)))
    {
      ERR(err);
    }
    if (AX_SUCCESS != err)
    {
      if (NULL != event)
      {
        ev_free_list_.push(event);
      }
      if (id != INVALID_ID)
      {
        free_sock(id);
      }
      else if (epfd >= 0)
      {
        close(epfd);
      }
    }
    else
    {
      epfd_ = epfd;
    }
    return err;
  }
  int create_event_fd() {
    int err = AX_SUCCESS;
    Id id = INVALID_ID;
    int evfd = -1;
    if ((evfd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)) < 0)
    {
      err = AX_EVENTFD_CREATE_ERR;
    }
    else if (AX_SUCCESS != (err = alloc_sock(id, evfd, SOCK_FLAG_REARM, 0)))
    {
      ERR(err);
    }
    if (AX_SUCCESS != err)
    {
      if (id != INVALID_ID)
      {
        free_sock(id);
      }
      else if (evfd >= 0)
      {
        close(evfd);
      }
    }
    else
    {
      evfd_ = evfd;
    }
    return err;
  }
  int listen(Server& server) {
    int err = AX_SUCCESS;
    Id id = INVALID_ID;
    struct sockaddr_in sin;
    int fd = -1;
    if (!server.is_valid())
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (epfd_ <= 0)
    {
      err = AX_NOT_INIT;
    }
    else if ((fd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
      err = AX_SOCK_CREATE_ERR;
    }
    else if (0 != bind(fd, (sockaddr*)make_sockaddr(&sin, server.ip_, server.port_), sizeof(sin)))
    {
      err = AX_SOCK_CREATE_ERR;
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
  int get(Server& server, SockHandler*& sock) {
    int err = AX_SUCCESS;
    int fd = -1;
    Id id = INVALID_ID;
    struct sockaddr_in sin;
    bool locked = false;
    sock = NULL;
    if (AX_SUCCESS != (err = out_sock_table_.lock(*(uint64_t*)(&server), (void*&)id)))
    {
      locked = true;
    }
    else if (INVALID_ID != id && AX_SUCCESS != sock_map_.fetch(id, (void*&)sock))
    {
      id = INVALID_ID;
      sock = NULL;
    }
    if (AX_SUCCESS != err || INVALID_ID != id)
    {}
    else if ((fd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
      err = AX_SOCK_CREATE_ERR;
    }
    else if (0 != make_fd_nonblocking(fd))
    {
      err = AX_FCNTL_ERR;
    }
    else if (0 != connect(fd, (sockaddr*)make_sockaddr(&sin, server.ip_, server.port_), sizeof(sin))
             && EINPROGRESS != errno)
    {
      err = AX_SOCK_CREATE_ERR;
    }
    else if (AX_SUCCESS != (err = alloc_sock(id, fd, SOCK_FLAG_CONNECTING, EPOLLIN | EPOLLOUT | EPOLLONESHOT)))
    {}
    if (locked)
    {
      out_sock_table_.unlock(*(uint64_t*)(&server), (void*)id);
    }
    if (NULL != sock || INVALID_ID == id)
    {}
    else if (AX_SUCCESS != (err = sock_map_.fetch(id, (void*&)sock)))
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
  int revert(SockHandler* sock, int handle_err) {
    int err = AX_SUCCESS;
    if (NULL == sock)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (AX_SUCCESS != (err = sock_map_.revert(sock->id_)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != handle_err)
    {
      err = free_sock(sock->id_);
    }
    return err;
  }
private:
  SpinQueue ev_free_list_;
  FutexQueue ev_queue_;
  CacheIndex out_sock_table_;
  IDMap sock_map_;
  SpinQueue rearm_queue_;
  int evfd_;
  int epfd_;
  SockHandler* handler_;
};

#endif /* __OB_AX_NIO_H__ */
