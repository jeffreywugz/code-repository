#ifndef __OB_AX_IO_SCHED_H__
#define __OB_AX_IO_SCHED_H__
#include "common.h"
#include "pcounter.h"
#include <sys/epoll.h>
#include <sys/eventfd.h>

inline bool set_if_bigger(uint64_t* val_ptr, uint64_t new_val, uint64_t seq_mask, uint64_t* ret_old_value) {
  bool set_succ = false;
  uint64_t new_seq = new_val & seq_mask;
  uint64_t old_val = 0;
  while((((old_val = AL(val_ptr)) & seq_mask) < new_seq) && !(set_succ = CAS(val_ptr, old_val, new_val)))
  {
    PAUSE();
  }
  if (NULL != ret_old_value)
  {
    *ret_old_value = old_val;
  }
  return set_succ;
}

struct ReadyFlag
{
  static const uint64_t SEQ_MASK = (~0UL)>>1;
  ReadyFlag(): seq_(1), flag_(0) {}
  ~ReadyFlag() {}
  uint64_t get_seq() { return FAA(&seq_, 1); }
  bool set_ready(uint64_t& seq) {
    uint64_t old_val = 0;
    seq = get_seq();
    bool succ = set_if_bigger(&flag_, (~SEQ_MASK) | seq, SEQ_MASK, &old_val);
    return succ && 0 == (old_val & ~SEQ_MASK);
  }
  bool set_finished(uint64_t seq) {
    return set_if_bigger(&flag_, seq, SEQ_MASK, NULL); }
  uint64_t seq_;
  uint64_t flag_;
};

inline struct epoll_event* make_epoll_event(epoll_event* event, uint32_t event_flag, Id id)
{
  if (NULL != event)
  {
    event->events = event_flag;
    event->data.u64 = id;
  }
  return event;
}

class IOSched;
struct Sock
{
  enum {EPOLL = 1, LISTEN = 2, NORMAL_TCP = 3, READY = 1U<<31};
  Sock(uint32_t flag): flag_(flag), id_(INVALID_ID), fd_(-1), sched_(NULL),
                       last_active_time_(get_us()), rbytes_(0), wbytes_(0) {}
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
  virtual int read() = 0;
  virtual int write() = 0;
  virtual bool kill() { return false; }
  char* repr(Printer& printer) {
    return printer.new_str("sock: id=%lx flag=%x ready4read=%lx ready4write=%lx fd=%d rb=%lu wb=%lu", id_, flag_, ready4read_.flag_, ready4write_.flag_, fd_, rbytes_, wbytes_);
  }
  void after_create() {
    MLOG(INFO, "create_sock: %s", ::repr(*this));
  }
  void before_destroy() {
    MLOG(INFO, "destroy_sock: %s %s", ::repr(*this), lbt());
  }
  Sock* set_fd(int fd) { fd_ = fd; return this; }
  void mark_active() { last_active_time_ = get_us(); }
  uint64_t flag_;
  Id id_;
  ReadyFlag ready4read_;
  ReadyFlag ready4write_;
  int fd_;
  IOSched* sched_;
  int64_t last_active_time_;
  uint64_t rbytes_;
  uint64_t wbytes_;
};

class IOSched
{
public:
  struct EpollSock: public Sock
  {
    EpollSock(): Sock(EPOLL), is_waiting_(false), evfd_(-1) {}
    virtual ~EpollSock() { destroy(); }
    int init(IOSched* sched) {
      int err = AX_SUCCESS;
      epoll_event event;
      Id id = INVALID_ID;
      if (NULL == sched)
      {
        err = AX_INVALID_ARGUMENT;
      }
      else if ((fd_ = epoll_create1(EPOLL_CLOEXEC)) < 0)
      {
        err = AX_EPOLL_CREATE_ERR;
      }
      else if (AX_SUCCESS != (err = sched->add(id, this, 0)))
      {
        ERR(err);
      }
      else if (AX_SUCCESS != (err = sched->set_ready_flag(id | IOSched::READ_FLAG)))
      {
        ERR(err);
      }
      else if ((evfd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)) < 0)
      {
        err = AX_EVENTFD_CREATE_ERR;
        ERR(err);
      }
      else if (0 != epoll_ctl(fd_, EPOLL_CTL_ADD, evfd_, make_epoll_event(&event, EPOLLIN, INVALID_ID)))
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
        else
        {
          destroy();
        }
      }
      return err;
    }
    int destroy() {
      if (evfd_ >= 0)
      {
        close(evfd_);
        evfd_ = -1;
      }
      Sock::destroy();
      return AX_SUCCESS;
    }
    int write() {
      return AX_EAGAIN;
    }
    int add(int fd, uint32_t flag, Id id) {
      int err = AX_SUCCESS;
      epoll_event event;
      if (fd == fd_)
      {}
      else if (0 != epoll_ctl(fd_, EPOLL_CTL_ADD, fd, make_epoll_event(&event, flag, id)))
      {
        err = AX_EPOLL_CTL_ERR;
      }
      return err;
    }
    int do_epoll_wait(int fd, epoll_event* events, int maxevents, int timeout) {
      int count = 0;
      PC_PROFILE(EPWAIT);
      AS(&is_waiting_, true);
      count = epoll_wait(fd, events, maxevents, timeout);
      AS(&is_waiting_, false);
      return count;
    }

    void wakeup() {
      if (AL(&is_waiting_))
      {
        eventfd_write(evfd_, 1);
      }
    }
    int read() {
      int err = AX_SUCCESS;
      int count = 0;
      int64_t timeout = 1000;
      epoll_event events[32];
      if (NULL == sched_)
      {
        err = AX_NOT_INIT;
      }
      else if (sched_->get_qsize() > 0)
      {}
      else if ((count = do_epoll_wait(fd_, events, arrlen(events), timeout)) < 0
               && EINTR != errno)
      {
        err = AX_EPOLL_WAIT_ERR;
        ERR(err);
      }
      else if (EINTR == errno)
      {}
      else if (count <= 0)
      {}
      else
      {
        for(int i = 0; i < count; i++)
        {
          handle_epoll_event(events[i].data.u64, events[i].events);
        }
      }
      return err;
    }
    void handle_epoll_event(Id id, uint32_t evmask) {
      if ((evmask & EPOLLERR) || (evmask & EPOLLHUP))
      {
        sched_->del(id);
        MLOG(INFO, "del sock: id=%lx receive EPOLLHUP", id);
      }
      else if (INVALID_ID == id)
      {
        uint64_t value = 0;
        eventfd_read(evfd_,  &value);
        PC_ADD(EVWAKE, 1);
      }
      else
      {
        if (evmask & EPOLLOUT)
        {
          sched_->set_ready_flag(id | IOSched::WRITE_FLAG);
        }
        if (evmask & EPOLLIN)
        {
          sched_->set_ready_flag(id | IOSched::READ_FLAG);
        }
      }
    }
    bool is_waiting_;
    int evfd_;
  };
public:
  IOSched(): killing_idx_(0), start_time_(0)  {}
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
    else if (AX_SUCCESS != (err = epoll_sock_.init(this)))
    {
      ERR(err);
    }
    else
    {
      start_time_ = get_us();
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
    for(int64_t idx = 0; idx < sock_map_.get_capacity(); idx++)
    {
      Id id = INVALID_ID;
      Sock* sock = NULL;
      if (AX_SUCCESS == sock_map_.fetch_by_idx(idx, id, (void*&)sock))
      {
        sock_map_.revert(id);
        del(id);
      }
    }
    io_queue_.destroy();
    sock_map_.destroy();
  }
  int64_t get_qsize () const { return io_queue_.get_size(); }
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
      PC_PROFILE(IO);
      if (WRITE_FLAG == (desc & (~ID_MASK)))
      {
        uint64_t seq = sock->ready4write_.get_seq();
        if (AX_EAGAIN == (err = sock->write()))
        {
          err = AX_SUCCESS;
          finished = sock->ready4write_.set_finished(seq);
        }
      }
      else
      {
        uint64_t seq = sock->ready4read_.get_seq();
        if (AX_EAGAIN == (err = sock->read()))
        {
          err = AX_SUCCESS;
          finished = sock->ready4read_.set_finished(seq);
        }
      }
      sock_map_.revert(id);
      if (finished)
      {}
      else if (AX_SUCCESS != (err = io_queue_.push((void*)desc)))
      {}
      else if (io_queue_.get_size() > 0)
      {
        epoll_sock_.wakeup();
      }
      if (AX_SUCCESS != err)
      {
        del(id);
        MLOG(INFO, "del: desc=%lx err=%d", desc, err);
      }
    }
    try_del_idle_sock();
    return err;
  }
  int add(Id& id, Sock* sock, uint32_t event_flag) {
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
    else if (AX_SUCCESS != (err = epoll_sock_.add(sock->fd_, event_flag, id)))
    {
      err = AX_EPOLL_CTL_ERR;
      ERR(err);
    }
    else
    {
      sock->id_ = id;
      sock->sched_ = this;
      sock->after_create();
      sock_map_.revert(id);
    }
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
  int fetch(Id id, Sock*& sock) {
    return sock_map_.fetch(id, (void*&)sock);
  }
  int revert(Id id) {
    return sock_map_.revert(id);
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
      uint64_t seq = 0;
      if (WRITE_FLAG == (flag & (~ID_MASK)))
      {
        ready = sock->ready4write_.set_ready(seq);
      }
      else
      {
        ready = sock->ready4read_.set_ready(seq);
      }
      sock_map_.revert(id);
      if (!ready)
      {}
      else if (AX_SUCCESS != (err = io_queue_.push((void*)flag)))
      {}
      else if (io_queue_.get_size() > 0)
      {
        epoll_sock_.wakeup();
      }
      if (AX_SUCCESS != err)
      {
        del(id);
      }
    }
    return err;
  }
  int try_del_idle_sock() {
    int err = AX_SUCCESS;
    uint64_t end_kill_idx = ((get_us() - start_time_)/1000000) * sock_map_.get_capacity();
    uint64_t idx = INVALID_ID;
    do {
      Id id = INVALID_ID;
      Sock* sock = NULL;
      idx = FAA(&killing_idx_, 1);
      if (AX_SUCCESS != (err = sock_map_.fetch_by_idx(idx, id, (void*&)sock)))
      {}
      else
      {
        bool can_kill = sock->kill();
        sock_map_.revert(id);
        if (can_kill)
        {
          del(id);
        }
      }
    } while(idx < end_kill_idx);
    return err;
  }
private:
  uint64_t killing_idx_;
  uint64_t start_time_;
  EpollSock epoll_sock_;
  FutexQueue io_queue_;
  IDMap sock_map_;
};

#endif /* __OB_AX_IO_SCHED_H__ */

