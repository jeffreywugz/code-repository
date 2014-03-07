#ifndef __OB_AX_NIO_H__
#define __OB_AX_NIO_H__
#include "common.h"
#include <sys/epoll.h>

bool set_if_bigger(uint64_t* val_ptr, uint64_t new_val, uint64_t seq_mask) {
  bool set_succ = false;
  uint64_t new_seq = new_val & seq_mask;
  uint64_t old_val = 0;
  while((((old_val = AL(val_ptr)) & seq_mask) < new_seq) && !(set_succ = CAS(value_ptr, old_val, new_val)))
  {
    PAUSE();
  }
  return set_succ;
}

struct ReadyFlag
{
  static uint64_t SEQ_MASK = (~0UL)>>1;
  ReadyFlag(): seq_(0), flag_(~SEQ_MASK) {}
  ~ReadyFlag() {}
  uint64_t get_seq() { return FAA(&seq_, 1); }
  bool set_ready() { return set_if_bigger(&flag_, (~SEQ_MASK) | get_seq(), SEQ_MASK);}
  bool set_finished(uint64_t seq) { return set_if_bigger(&flag_, seq, SEQ_MASK); }
  uint64_t seq_;
  uint64_t flag_;
};

struct Sock
{
  Sock(): id_(INVALID_ID), fd_(-1) {}
  virtual ~Sock() {}
  virtual int clone(Sock*& other) = 0;
  virtual int destroy() = 0;
  virtual int read() = 0;
  virtual int write() = 0;
  char* repr(Printer& printer) {
    return printer.new_str("sock: id=%lx ready4read=%lx ready4write=%lx fd=%d", id_, ready4read_.flag_, ready4write_.flag_, fd_);
  }
  void after_create() {
    MLOG(INFO, "create_sock: %s", repr(*this));
  }
  void before_destroy() {
    MLOG(INFO, "destroy_sock: %s", repr(*this));
  }
  Id id_;
  ReadyFlag ready4read_;
  ReadyFlag ready4write_;
  int fd_;
};

class IOSchedule
{
public:
  IOSchedule() {}
  ~IOSchedule() { destroy(); }
  static uint64_t ID_MASK = (~0UL) >> 1;
  static uint64_t WRITE_FLAG = 1UL<<63;
  static uint64_t READ_FLAG = 0UL;
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
    else if (AX_SUCCESS != (err = sock_map_.fetch((id = (desc & ID_MASK)), sock)))
    {
      ERR(err);
    }
    else
    {
      bool finished = false;
      if (WRITE_FALG == (desc & (~ID_MASK)))
      {
        uint64_t seq = sock->ready4write_.get_seq();
        if (AX_EAGAIN == (err = sock->handle_write()))
        {
          err = AX_SUCCESS;
          finished = sock->ready4write_.set_finished(seq);
        }
      }
      else
      {
        uint64_t seq = sock->ready4read_.get_seq();
        if (AX_EAGAIN == (err = sock->handle_read()))
        {
          err = AX_SUCCESS;
          finished = sock->ready4read_.set_finished(seq);
        }
      }
      sock_map_.revert(id);
      if (AX_SUCCESS != err)
      {
        del_sock(id);
      }
      if (!finished)
      {
        err = io_queue_.push(desc);
      }
    }
    return err;
  }
  int add_sock(Id& id, Sock* sock) {
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
    if (AX_SUCCESS != err)
    {}
    else if (AX_SUCCESS != (io_queue_.push(id | WRITE_FLAG)))
    {}
    else if (AX_SUCCESS != (io_queue_.push(id | READ_FLAG)))
    {}
    if (INVALID_ID != id)
    {
      if (AX_SUCCESS != err)
      {
        del_sock(id);
        id = INVALID_ID;
      }
    }
    return err;
  }
  int del_sock(Id id) {
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
private:
  FutexQueue io_queue_;
  IDMap sock_map_;
};

#endif /* __OB_AX_NIO_H__ */
