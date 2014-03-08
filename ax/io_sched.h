#ifndef __OB_AX_IO_SCHED_H__
#define __OB_AX_IO_SCHED_H__
#include "common.h"

inline bool set_if_bigger(uint64_t* val_ptr, uint64_t new_val, uint64_t seq_mask) {
  bool set_succ = false;
  uint64_t new_seq = new_val & seq_mask;
  uint64_t old_val = 0;
  while((((old_val = AL(val_ptr)) & seq_mask) < new_seq) && !(set_succ = CAS(val_ptr, old_val, new_val)))
  {
    PAUSE();
  }
  return set_succ;
}

struct ReadyFlag
{
  static const uint64_t SEQ_MASK = (~0UL)>>1;
  ReadyFlag(): seq_(1), flag_(0) {}
  ~ReadyFlag() {}
  uint64_t get_seq() { return FAA(&seq_, 1); }
  bool set_ready() { return set_if_bigger(&flag_, (~SEQ_MASK) | get_seq(), SEQ_MASK);}
  bool set_finished(uint64_t seq) { return set_if_bigger(&flag_, seq, SEQ_MASK); }
  uint64_t seq_;
  uint64_t flag_;
};

class IOSched;
struct Sock
{
  enum {EPOLL = 1, LISTEN = 2, NORMAL = 3};
  Sock(uint64_t type): type_(type), id_(INVALID_ID), fd_(-1) {}
  virtual ~Sock() {}
  virtual int clone(Sock*& sock) {
    sock = NULL;
    return AX_NOT_SUPPORT;
  }
  virtual int destroy() {
    int err = AX_SUCCESS;
    before_destroy();
    if (fd_ >= 0)
    {
      close(fd_);
      fd_ = -1;
    }
    return err;
  }
  virtual int read(IOSched* sched) = 0;
  virtual int write(IOSched* sched) = 0;
  char* repr(Printer& printer) {
    return printer.new_str("sock: id=%lx type=%lx ready4read=%lx ready4write=%lx fd=%d", id_, type_, ready4read_.flag_, ready4write_.flag_, fd_);
  }
  void after_create() {
    MLOG(INFO, "create_sock: %s", ::repr(*this));
  }
  void before_destroy() {
    MLOG(INFO, "destroy_sock: %s %s", ::repr(*this), lbt());
  }
  Sock* set_fd(int fd) { fd_ = fd; return this; }
  uint64_t type_;
  Id id_;
  ReadyFlag ready4read_;
  ReadyFlag ready4write_;
  int fd_;
};

class IOSched
{
public:
  IOSched() {}
  ~IOSched() { destroy(); }
  static const uint64_t ID_MASK = (~0UL) >> 1;
  static const uint64_t WRITE_FLAG = 1UL<<63;
  static const uint64_t READ_FLAG = 0UL;
public:
  int init(int64_t capacity, char* buf) {
    int err = AX_SUCCESS;
    MemChunkCutter cutter(calc_mem_usage(capacity), buf);
    if (capacity <= 0 || !is2n(capacity) || NULL == buf)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (AX_SUCCESS != (err = init_container(io_queue_, capacity, cutter)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = init_container(sock_map_, capacity, cutter)))
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
    return io_queue_.calc_mem_usage(capacity) + sock_map_.calc_mem_usage(capacity);
  }
  void destroy() {
    io_queue_.destroy();
    sock_map_.destroy();
  }
  int sched(int64_t timeout_us) {
    int err = AX_SUCCESS;
    Id desc = INVALID_ID;
    Id id = INVALID_ID;
    Sock* sock = NULL;
    if (AX_SUCCESS != (err = io_queue_.pop((void*&)desc, timeout_us)))
    {}
    else if (AX_SUCCESS != (err = sock_map_.fetch((id = (desc & ID_MASK)), (void*&)sock)))
    {
      MLOG(WARN, "io_sched: desc=%lx is deleted", desc);
    }
    else
    {
      bool finished = false;
      if (WRITE_FLAG == (desc & (~ID_MASK)))
      {
        uint64_t seq = sock->ready4write_.get_seq();
        if (AX_EAGAIN == (err = sock->write(this)))
        {
          err = AX_SUCCESS;
          finished = sock->ready4write_.set_finished(seq);
        }
      }
      else
      {
        uint64_t seq = sock->ready4read_.get_seq();
        if (AX_EAGAIN == (err = sock->read(this)))
        {
          err = AX_SUCCESS;
          finished = sock->ready4read_.set_finished(seq);
        }
      }
      sock_map_.revert(id);
      if (AX_SUCCESS != err)
      {
        del(id);
        MLOG(INFO, "del: desc=%lx err=%d", desc, err);
      }
      if (!finished)
      {
        err = io_queue_.push((void*)desc);
      }
    }
    return err;
  }
  int add(Id& id, Sock* sock) {
    int err = AX_SUCCESS;
    if (NULL == sock)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (AX_SUCCESS != (err = sock_map_.add(id, (void*)sock)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = sock_map_.fetch(id, (void*&)sock)))
    {
      id = INVALID_ID;
      ERR(err);
    }
    else
    {
      sock->id_ = id;
      sock->after_create();
      sock_map_.revert(id);
    }
    // if (AX_SUCCESS != err)
    // {}
    // else if (AX_SUCCESS != (io_queue_.push((void*)(id | WRITE_FLAG))))
    // {}
    // else if (AX_SUCCESS != (io_queue_.push((void*)(id | READ_FLAG))))
    // {}
    if (INVALID_ID != id)
    {
      if (AX_SUCCESS != err)
      {
        del(id);
        id = INVALID_ID;
      }
    }
    return err;
  }
  int del(Id id) {
    int err = AX_SUCCESS;
    Sock* sock = NULL;
    if (AX_SUCCESS != (err = sock_map_.del(id, (void*&)sock)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = sock->destroy()))
    {
      ERR(err);
    }
    return err;
  }
  int set_ready_flag(Id flag) {
    int err = AX_SUCCESS;
    Sock* sock = NULL;
    Id id = flag & ID_MASK;
    if (AX_SUCCESS != (err = sock_map_.fetch(id, (void*&)sock)))
    {
      ERR(err);
    }
    else
    {
      bool ready = false;
      if (WRITE_FLAG == (id & (~ID_MASK)))
      {
        ready = sock->ready4write_.set_ready();
      }
      else
      {
        ready = sock->ready4read_.set_ready();
      }
      sock_map_.revert(id);
      if (!ready)
      {}
      else if (AX_SUCCESS != (err = io_queue_.push((void*)flag)))
      {
        del(id);
      }
    }
    return err;
  }
private:
  FutexQueue io_queue_;
  IDMap sock_map_;
};

#include <sys/epoll.h>
struct EpollSock: public Sock
{
  EpollSock(): Sock(EPOLL) {}
  virtual ~EpollSock() { destroy(); }
  int init(IOSched* sched) {
    int err = AX_SUCCESS;
    Id id = INVALID_ID;
    if (NULL == sched)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if ((fd_ = epoll_create1(EPOLL_CLOEXEC)) < 0)
    {
      err = AX_EPOLL_CREATE_ERR;
    }
    else if (AX_SUCCESS != (err = sched->add(id, this)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = sched->set_ready_flag(id | IOSched::READ_FLAG)))
    {
      ERR(err);
    }
    if (AX_SUCCESS != err)
    {
      destroy();
    }
    return err;
  }
  int write(IOSched* sched) {
    return AX_EAGAIN;
  }
  int read(IOSched* sched) {
    int err = AX_SUCCESS;
    int count = 0;
    int64_t timeout = 1000;
    epoll_event events[32];
    if ((count = epoll_wait(fd_, events, arrlen(events), timeout)) < 0
        && EINTR != errno)
    {
      err = AX_EPOLL_WAIT_ERR;
      ERR(err);
    }
    else if (EINTR == errno)
    {}
    else if (count > 0)
    {
      MLOG(INFO, "epoll_wait: count=%d", count);
      for(int i = 0; i < count; i++)
      {
        uint32_t evmask = events[i].events;
        Id id = events[i].data.u64;
        if ((evmask & EPOLLERR) || (evmask & EPOLLHUP))
        {
          sched->del(id);
          MLOG(INFO, "del sock: id=%lx", id);
        }
        else
        {
          if (evmask & EPOLLOUT)
          {
            sched->set_ready_flag(id | IOSched::WRITE_FLAG);
          }
          if (evmask & EPOLLIN)
          {
            sched->set_ready_flag(id | IOSched::READ_FLAG);
          }
        }
      }
    }
    return err;
  }
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

struct ListenSock: public Sock
{
  ListenSock(): Sock(LISTEN), epfd_(-1), sock_(NULL) {}
  virtual ~ListenSock() { destroy(); }
  int init(int epfd, Sock* sock, struct sockaddr_in* addr, IOSched* sched) {
    int err = AX_SUCCESS;
    Id id = INVALID_ID;
    int backlog = 128;
    struct epoll_event event;
    if (epfd < 0 || NULL == sock || NULL == addr || NULL == sched)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if ((fd_ = ::socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
      err = AX_SOCK_CREATE_ERR;
      ERR(err);
    }
    else if (0 != ::bind(fd_, (sockaddr*)addr, sizeof(*addr)))
    {
      err = AX_SOCK_BIND_ERR;
      ERR(err);
    }
    else if (0 != ::listen(fd_, backlog))
    {
      err = AX_SOCK_LISTEN_ERR;
    }
    else if (0 != make_fd_nonblocking(fd_))
    {
      err = AX_FCNTL_ERR;
    }
    else if (AX_SUCCESS != (err = sched->add(id, this)))
    {
      ERR(err);
    }
    else if (0 != epoll_ctl(epfd, EPOLL_CTL_ADD, fd_, make_epoll_event(&event, EPOLLET | EPOLLIN, id)))
    {
      err = AX_EPOLL_CTL_ERR;
      ERR(err);
    }
    else
    {
      epfd_ = epfd;
      sock_ = sock;
    }
    if (AX_SUCCESS != err)
    {
      if (INVALID_ID != id)
      {
        sched->del(id);
      }
      else
      {
        destroy();
      }
    }
    return err;
  }
  int destroy() {
    int err = AX_SUCCESS;
    epfd_ = -1;
    sock_ = NULL;
    Sock::destroy();
    return err;
  }
  int write(IOSched* sched) {
    return AX_EAGAIN;
  }
  int read(IOSched* sched) {
    int err = AX_SUCCESS;
    Sock* sock = NULL;
    epoll_event event;
    Id id = INVALID_ID;
    int fd = -1;
    MLOG(INFO, "try accpet incomming connection,listen_fd=%d", fd_);
    if ((fd = accept(fd_, NULL, NULL)) < 0)
    {
      if (EINTR == errno)
      {}
      else if (EAGAIN == errno || EWOULDBLOCK == errno)
      {
        err = AX_EAGAIN;
      }
      else
      {
        err = AX_SOCK_ACCEPT_ERR;
      }
    }
    else if (0 != make_fd_nonblocking(fd))
    {
      err = AX_FCNTL_ERR;
    }
    else if (epfd_ < 0 || NULL == sock_)
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS != (err = sock_->clone(sock)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = (sched->add(id, sock->set_fd(fd)))))
    {
      ERR(err);
    }
    else if (0 != epoll_ctl(epfd_, EPOLL_CTL_ADD, fd, make_epoll_event(&event, EPOLLET | EPOLLIN | EPOLLOUT, id)))
    {
      err = AX_EPOLL_CTL_ERR;
      ERR(err);
    }
    if (AX_SUCCESS != err)
    {
      if (INVALID_ID != id)
      {
        sched->del(id);
      }
      else if (NULL != sock)
      {
        sock->destroy();
      }
      else if (fd >= 0)
      {
        close(fd);
      }
    }
    return err;
  }
  int epfd_;
  Sock* sock_;
};

#endif /* __OB_AX_IO_SCHED_H__ */

