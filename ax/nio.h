#ifndef __OB_AX_NIO_H__
#define __OB_AX_NIO_H__
enum {SOCK_FLAG_EPOLL = 1, SOCK_FLAG_REARM = 2, SOCK_FLAG_LISTEN = 4};
struct SockHandler
{
  typedef int (*clone_func_t)(SockHandler* self, SockHandler*& other);
  typedef int (*destroy_func_t)(SockHandler* self);
  typedef int (*read_func_t)(SockHandler* self);
  typedef int (*write_func_t)(SockHandler* self);
  clone_func_t clone_func_;
  destroy_func_t destroy_func_;
  read_func_t read_func_;
  write_func_t write_func_;
  SockHandler(): clone_func_(NULL), destroy_func_(NULL), read_func_(NULL), write_func_(NULL),
                 id_(INVALID_ID), flag_(), fd_(-1), epfd_(-1) {}
  ~SockHandler() {}
  SockHandler* set_args(id_t id, int flag, int fd, int epfd, uint32_t events_flag) {
    id_ = id;
    flag_ = flag;
    fd_ = fd;
    epfd_ = epfd;
    event.data.u64 = id;
    event.events = events_flag;
    return this;
  }
  bool try_lock() { return lock_.try_lock(); }
  void unlock() { return lock_.unlock(); }
  bool is_epoll_fd() const { return flag_ & SOCK_FLAG_EPOLL; }
  bool is_listen_fd() const { return flag_ & SOCK_FLAG_LISTEN; }
  int clone(SockHandler*& other) { return NULL == clone_func_? AX_CALLBACK_NOT_SET: clone_func_(this, other); }
  int destroy() {
    int err = (NULL == destroy_func_? AX_CALLBACK_NOT_SET: destroy_func_(this));
    if (fd_ >= 0)
    {
      close(fd_);
    }
    return err;
  }
  int read() { return NULL == read_func_? AX_CALLBACK_NOT_SET: read_func_(this); }
  int write() { return NULL == write_func_? AX_CALLBACK_NOT_SET: write_func_(this); }
  SpinLock lock_;
  id_t id_;
  int flag_;
  int fd_;
  int epfd_;
  epoll_event event_;
};

int make_fd_non_blocking(int fd)
{
  int err = 0;
  int flags = 0;
  if ((flags = fcntl(fd, F_GETFL, 0)) < 0 || fcntl(fd, F_SETFL, flags|O_NONBLOCK) < 0)
  {
    err = -errno;
  }
  return err;
}

struct sockaddr_in* make_sockaddr(struct sockaddr_in* sin, in_addr_t ip, int port)
{
  if (NULL != sin)
  {
    sin->sin_port = htons(port);
    sin->sin_addr.s_addr = ip;
    sin->sin_family = AF_INET;
  }
  return sin;
}
int make_inet_socket(int type, in_addr_t ip, int port)
{
  int err = 0;
  int fd = -1;
  struct sockaddr_in sin;
  if ((fd = socket(AF_INET, type, 0)) < 0
      || bind(fd, make_sockaddr(&sin, ip, port),sizeof(sin)) != 0)
  {
    err = -errno;
  }
  if (fd > 0 && 0 != err)
  {
    close(fd);
  }
  return 0 == err? fd: -1;
}

struct epoll_event* set_epoll_event(epoll_event* event, uint32_t event_flag, id_t id)
{
  if (NULL != event)
  {
    event.events = event_flag;
    event.data.u64 = id;
  }
  return event;
}

class Nio
{
public:
  Nio(): evfd_(-1), epfd_(-1), handler_(NULL) {}
  ~Nio() { destroy(); }
public:
  int init(SockHandler* handler, int64_t capacity, void* buf) {
    int err = AX_SUCCESS;
    MemChunkCutter cutter(calc_mem_usage(capacity), buf);
    if (capacity <= 0 || !is2n(capacity) || NULL == handler || NULL == buf)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL == handler_)
    {
      err = AX_INIT_TWICE;
    }
    else if (AX_SUCCESS != (init_container(ev_free_list_, capacity, cutter)))
    {}
    else if (AX_SUCCESS != (init_container(ev_queue_, capacity, cutter)))
    {}
    else if (AX_SUCCESS != (init_container(out_sock_table_, capacity, cutter)))
    {}
    else if (AX_SUCCESS != (init_container(sock_map_, capacity, cutter)))
    {}
    else
    {
      handler_ = handler;
    }
    if (AX_SUCCESS == err)
    {
      err = create_epoll();
    }
    if (AX_SUCCESS != err)
    {
      do_destroy();
    }
    return err;
  }
  int64_t calc_mem_usage(int64_t capacity) {
    return ev_free_list_.calc_mem_usage(capacity) + ev_queue_.calc_mem_usage() + out_sock_table_.calc_mem_usage() + sock_map_.calc_mem_usage();
  }
  void destroy() {
    ev_free_list_.destroy();
    ev_queue_.destroy();
    out_sock_table_.destroy();
    sock_array_.destroy();
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
  int do_loop(int64_t timeou_us) {
    int err = AX_SUCCESS;
    Event* event = NULL;
    id_t id = INVALID_ID;
    if (AX_SUCCESS != (ev_queue_.pop(event, timeout_us)))
    {}
    else if (AX_SUCCESS != (err = sock_map_.fetch((id = event.data.u64), handler)))
    {}
    else if (AX_SUCCESS != (err = handle_event(handler, event)))
    {
      sock_map_.revert(id);
      free_sock(id);
    }
    else
    {
      epoll_ctl(handler_->epfd_, EPOLL_CTL_MOD, handler->fd_, handler->event_);
      sock_map_.revert(id);
    }
    if (NULL != event)
    {
      err = ev_free_list_.push(event);
    }
    return err;
  }
protected:
  int handle_event(SockHandler* handler, epoll_event* event) {
    int err = AX_SUCCESS;
    int timeout = 10 * 1000;
    if (NULL == handler)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (!handler->try_lock())
    {}
    else if (is_epoll_fd(handler))
    {
      epoll_event events[32];
      int count = 0;
      if (AX_SUCCESS != (err = handle_rearm_queue()))
      {}
      else if ((count = epoll_wait(handler->fd_, events, array_len(events) - 1, timeout)) < 0)
      {
        err = AX_EPOLL_WAIT_ERR;
      }
      else
      {
        events[count++] = *event;
        for(int i = 0; i < count; i++)
        {
          epoll_event* event = NULL;
          while(AX_SUCCESS != ev_free_list_.pop(event))
            ;
          *event = events[i];
          while(AX_SUCCESS != ev_queue_.push(event))
            ;
          }
        }
      }
    }
    else if (is_listen_fd(handler))
    {
      int fd = -1;
      id_t id = INVALID_ID;
      if ((fd= accept(handler->fd_, NULL, NULL)) < 0)
      {
        if (errno != EAGAIN && errno != EWOULDBLOCK)
        {
          err = AX_ACCEPT_ERR;
        }
      }
      else if (AX_SUCCESS != (err = (alloc_sock(id, fd, 0, EPOLLIN | EPOLLONESHOT))))
      {}
    }
    else
    {
      if ((event & EPOLLOUT)
          && (AX_SUCCESS != (err = handler->write())))
      {}
      else if ((event & EPOLLIN)
               && (AX_SUCCESS != (err = handler->read())))
      {}
    }
    return err;
  }
  int handle_rearm_queue() {
    int err = AX_SUCCESS;
    id_t id = INVALID_ID;
    while(AX_SUCCESS == err)
    {
      if (AX_SUCCESS != (err = rearm_queue_.pop(id)))
      {}
      else if (AX_SUCCESS != (tmperr = sock_map_.fetch(id, sock)))
      {}
      else
      {
        if (0 != epoll_ctl(handler_->epfd_, EPOLL_CTL_MOD, handler->fd_, set_epoll_event(&event, id, sock->get_event_flag())))
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
  int create_sock_handler(SockHandler*& handler) {
    return NULL == handler? AX_NOT_INIT: handler_->clone(handler);
  }
  int destroy_sock_handler(SockHandler* handler) {
    return NULL == handler? AX_INVALID_ARGUMENT: handler->destroy();
  }

  int alloc_sock(id_t& id, int fd, int flag, uint32_t events_flag) {
    int err = AX_SUCCESS;
    epoll_event event;
    SockHandler* handler = NULL;
    if (AX_SUCCESS != (err = create_sock_handler(handler)))
    {}
    else if (AX_SUCCESS != (err = sock_map_.add(id, handler->set_args(INVALID_ID, flag, fd, epfd_, events_flag))))
    {}
    else if (AX_SUCCESS != (err = sock_map_.fetch(id, handler)))
    {}
    else
    {
      handler->id_ = id;
      sock_map_.revert(id);
      if (0 != epoll_ctl(epfd_, EPOLL_CTL_ADD, fd, set_epoll_event_struct(&event, id, events_flag)))
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
  int free_sock(id_t id) {
    int err = AX_SUCCESS;
    SockHandler* handler = NULL;
    if (AX_SUCCESS != (err = sock_map_.del(id, handler)))
    {}
    else if (AX_SUCCESS != (err = destroy_sock_handler(handler)))
    {}
    return err;
  }
  int create_epoll_fd() {
    int err = AX_SUCCESS;
    id_t id = INVALID_ID;
    int epfd = -1;
    struct epoll_event* event = NULL;
    if ((epfd = epoll_create1(EPOLL_CLOEXEC)) < 0)
    {
      err = AX_EPOLL_CREATE_ERR;
    }
    else if (AX_SUCCESS != (err = alloc_sock(id, epfd, SOCK_FLAG_EPOLL, 0)))
    {}
    else if (AX_SUCCESS != (err = ev_free_list_.pop(event)))
    {}
    else if (AX_SUCCESS != (err = ev_queue_.push(event)))
    {}
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
    id_t id = INVALID_ID;
    int evfd = -1;
    struct epoll_event* event = NULL;
    if ((evfd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)) < 0)
    {
      err = AX_EFD_CREATE_ERR;
    }
    else if (AX_SUCCESS != (err = alloc_sock(id, evfd, SOCK_FLAG_EVENTFD, 0)))
    {}
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
    id_t id = INVALID_ID;
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
    else if ((fd = socket(AF_INET, type, 0)) < 0)
    {
      err = AX_SOCK_CREATE_ERR;
    }
    else if (0 != bind(fd, make_sockaddr(&sin, server.ip_, server.port_), sizeof(sin)))
    {
      err = AX_SOCK_CREATE_ERR;
    }
    else if (0 != make_fd_nonblocking(fd))
    {
      err = AX_FCNTL_ERR;
    }
    else if (AX_SUCCESS != (err = alloc_sock(id, fd, SOCK_TYPE_LISTEN, EPOLLIN | EPOLLONESHOT)))
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
  int get(Server& server, SockHandler*& sock) {
    int err = AX_SUCCESS;
    id_t id = INVALID_ID;
    struct sockaddr_in sin;
    bool locked = false;
    sock = NULL;
    if (AX_SUCCESS != (err = out_sock_table_.lock(server, id)))
    {
      locked = true;
    }
    else if (INVALID_ID != id && AX_SUCCESS != sock_map_.fetch(id, sock))
    {
      id = INVALID_ID;
      sock = NULL;
    }
    if (OB_SUCCESS != err || INVALID_ID != id)
    {}
    else if ((fd = socket(AF_INET, type, 0)) < 0)
    {
      err = AX_CREATE_SOCK_ERR;
    }
    else if (0 != make_fd_nonblocking(fd))
    {
      err = AX_FCNTL_ERR;
    }
    else if (0 != connect(fd, make_sockaddr(&sin, server.ip_, server.port_), sizeof(sin))
             && EINPROGRESS != errno)
    {
      err = AX_CREATE_SOCK_ERR;
    }
    else if (AX_SUCCESS != (err = alloc_sock(id, fd, 0, EPOLLIN | EPOLLOUT | EPOLLONESHOT)))
    {}
    if (locked)
    {
      out_sock_table_.unlock(server, id);
    }
    if (NULL != sock || INVALID_ID == id)
    {}
    else if (AX_SUCCESS != (err = sock_map_.fetch(id, sock)))
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
    if (NULL == handler)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (AX_SUCCESS != (err = sock_map_.revert(handler->id)))
    {}
    else if (AX_SUCCESS != handle_err)
    {
      err = free_sock(handler->id_);
    }
    return err;
  }
private:
  SpinQueue ev_free_list_;
  FutexQueue ev_queue_;
  CacheIndex out_sock_table_;
  IdMap sock_map_;
  SpinQueue rearm_queue_;
  int evfd_;
  int epfd_;
  SockHandler* handler_;
};

#endif /* __OB_AX_NIO_H__ */
