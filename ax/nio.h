#ifndef __OB_AX_NIO_H__
#define __OB_AX_NIO_H__
#include "io_sched.h"
#include "link_queue.h"

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

struct ListenSock: public Sock
{
  ListenSock(): Sock(LISTEN), sock_(NULL) {}
  virtual ~ListenSock() { destroy(); }
  int init(IOSched* sched, Sock* sock, struct sockaddr_in* addr) {
    int err = AX_SUCCESS;
    Id id = INVALID_ID;
    int backlog = 128;
    if (NULL == sched || NULL == sock || NULL == addr)
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
    else if (AX_SUCCESS != (err = sched->add(id, this, EPOLLET | EPOLLIN)))
    {
      ERR(err);
    }
    else
    {
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
  int clone_sock(Sock*& sock) {
    int err = AX_SUCCESS;
    if (NULL == sock_)
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS != (err = sock_->clone(sock)))
    {}
    else
    {
      sock->flag_ |= READY;
    }
    return err;
  }
  int write() {
    return AX_EAGAIN;
  }
  int read() {
    int err = AX_SUCCESS;
    Sock* sock = NULL;
    Id id = INVALID_ID;
    int fd = -1;
    MLOG(INFO, "try accpet incomming connection,listen_fd=%d", fd_);
    if (NULL == sched_)
    {
      err = AX_NOT_INIT;
    }
    else if ((fd = accept(fd_, NULL, NULL)) < 0)
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
    else if (AX_SUCCESS != (err = clone_sock(sock)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = (sched_->add(id, sock->set_fd(fd), EPOLLET | EPOLLIN | EPOLLOUT))))
    {
      ERR(err);
    }
    if (AX_SUCCESS != err)
    {
      if (INVALID_ID != id)
      {
        sched_->del(id);
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


class OutSockCache
{
public:
  OutSockCache(): sched_(NULL), sock_(NULL) {}
  ~OutSockCache() {}
public:
  int init(IOSched* sched, Sock* sock, int64_t capacity, char* buf) {
    int err = AX_SUCCESS;
    if (NULL == sched || NULL == sock || NULL == buf || capacity <= 0 || !is2n(capacity))
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (AX_SUCCESS != (err = cache_.init(capacity, buf)))
    {}
    else
    {
      sched_ = sched;
      sock_ = sock;
    }
    return err;
  }
  int64_t calc_mem_usage(int64_t capacity) {
    return cache_.calc_mem_usage(capacity);
  }
  int fetch(Server server, Sock*& sock) {
    int err = AX_SUCCESS;
    struct sockaddr_in sin;
    Id id = INVALID_ID;
    bool locked = false;
    sock = NULL;
    if (!server.is_valid())
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL == sock_ || NULL == sched_)
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS != (err = cache_.lock(*(uint64_t*)(&server), (void*&)id)))
    {
      locked = true;
    }
    else if (INVALID_ID != id && AX_SUCCESS == (err = sched_->fetch(id, (Sock*&)sock)))
    {}
    else if (AX_SUCCESS != (err = connect(make_sockaddr(&sin, server.ip_, server.port_), sock)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != (err = sched_->add(id, sock, EPOLLET | EPOLLIN | EPOLLOUT)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != sched_->fetch(id, (Sock*&)sock))
    {
      sock = NULL;
    }
    if (AX_SUCCESS != err)
    {
      id = INVALID_ID;
      if (NULL != sock)
      {
        sock->destroy();
      }
    }
    if (locked)
    {
      cache_.unlock(*(uint64_t*)(&server), (void*)id);
    }
    return err;
  }
  int revert(Sock* sock, int handle_err) {
    int err = AX_SUCCESS;
    if (NULL == sock)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL == sched_)
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS != (err = sched_->revert(sock->id_)))
    {
      ERR(err);
    }
    else if (AX_SUCCESS != handle_err)
    {
      err = sched_->del(sock->id_);
    }
    return err;
  }
private:
  int connect(struct sockaddr_in* addr, Sock*& sock) {
    int err = AX_SUCCESS;
    int fd = -1;
    if (NULL == addr)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL == sock_)
    {
      err = AX_NOT_INIT;
    }
    else if ((fd = ::socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
      err = AX_SOCK_CREATE_ERR;
    }
    else if (0 != make_fd_nonblocking(fd))
    {
      err = AX_FCNTL_ERR;
    }
    else if (0 != ::connect(fd, (sockaddr*)addr, sizeof(*addr))
             && EINPROGRESS != errno)
    {
      err = AX_SOCK_CREATE_ERR;
    }
    else if (AX_SUCCESS != (err = sock_->clone(sock)))
    {
      ERR(err);
    }
    else
    {
      sock->set_fd(fd);
    }
    if (AX_SUCCESS != err)
    {
      if (NULL != sock)
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
private:
  IOSched* sched_;
  Sock* sock_;
  CacheIndex cache_;
};

struct Packet: public LinkNode
{
  Packet(): len_(0), checksum_(0) {}
  ~Packet() {}
  bool is_done(int64_t offset) {
    return get_remain(offset) <= 0;
  }
  int64_t get_remain(int64_t offset) {
    return (payload_ + len_ - header_) - offset;
  }
  bool check_checksum() {
    return true;
  }
  int set_buf(char* buf, int64_t len) {
    int err = AX_SUCCESS;
    if (NULL == buf || len <= 0)
    {
      len_ = 0;
    }
    else
    {
      len_ = len;
      memcpy(payload_, buf, len);
    }
    return err;
  }
  int get_buf(int64_t offset, char*& buf, int64_t& len) {
    int err = AX_SUCCESS;
    if (offset < 0)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if ((len = get_remain(offset)) <= 0)
    {
      err = AX_BUF_OVERFLOW;
    }
    else
    {
      buf = this->header_ + offset;
    }
    return err;
  }
  int clone(Packet* other) {
    int err = AX_SUCCESS;
    if (NULL == other)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else
    {
      memcpy(other->header_, this->header_, get_remain(0));
    }
    return err;
  }
  char header_[0];
  uint32_t len_;
  uint32_t checksum_;
  char payload_[1024];
};

class PacketHandler
{
public:
  PacketHandler() {}
  virtual ~PacketHandler() {}
public:
  virtual int alloc_packet(Packet*& pkt) = 0;
  virtual int free_packet(Packet* pkt) = 0;
  virtual int handle_packet(Id id, Packet* pkt) = 0;
};

struct TcpSock: public Sock
{
  TcpSock(): Sock(NORMAL_TCP), pkt_handler_(NULL), writen_offset_(0), writing_pkt_(NULL), read_offset_(0), reading_pkt_(NULL) {}
  virtual ~TcpSock() {}
  int init(IOSched* sched, PacketHandler* handler) {
    int err = AX_SUCCESS;
    sched_ = sched;
    pkt_handler_ = handler;
    return err;
  }
  int clone(Sock*& other) {
    int err = AX_SUCCESS;
    TcpSock* new_sock = new TcpSock();
    new_sock->pkt_handler_ = pkt_handler_;
    other = new_sock;
    return err;
  }
  int destroy() {
    int err = AX_SUCCESS;
    Sock::destroy();
    delete this;
    return err;
  }
  virtual bool kill() {
    int64_t idle_timeout = 5 * 1000000;
    return last_active_time_ + idle_timeout < get_us();
  }
  int read() {
    int err = AX_SUCCESS;;
    char* buf = NULL;
    int64_t len = 0;
    ssize_t rbytes = 0;
    mark_active();
    if (0 == (flag_ & READY))
    {}
    else
    {
      while(AX_SUCCESS == err && AX_SUCCESS == (err = get_read_buf(buf, len)))
      {
        if ((rbytes = ::read(fd_, buf, len)) < 0)
        {
          if (errno == EINTR)
          {}
          else if (EAGAIN == errno || EWOULDBLOCK == errno)
          {
            err = AX_EAGAIN;
          }
        }
        else
        {
          if (rbytes == 0)
          {
            err = AX_SOCK_HUP;
          }
          else if (rbytes < len)
          {
            err = AX_EAGAIN;
          }
          //MLOG(INFO, "fd=%d read: %.*s rbytes=%ld err=%d", fd_, rbytes, buf, rbytes, err);
          read_done(rbytes);
        }
      }
    }
    return err;
  }
  int write() {
    int err = AX_SUCCESS;
    char* buf = NULL;
    int64_t len = 0;
    int64_t wbytes = 0;
    mark_active();
    if (0 == (flag_ & READY))
    {
      int sys_err = 0;
      socklen_t errlen = sizeof(sys_err);
      if (0 != getsockopt(fd_, SOL_SOCKET, SO_ERROR, &sys_err, &errlen))
      {
        err = AX_GET_SOCKOPT_ERR;
        MLOG(ERROR, "connect error: fd=%d err=%d", fd_, sys_err);
      }
      else
      {
        flag_ |= READY;
      }
    }
    if (AX_SUCCESS == err)
    {
      while(AX_SUCCESS == (err = get_write_buf(buf, len)))
      {
        if ((wbytes = ::write(fd_, buf, len)) < 0)
        {
          if (errno == EINTR)
          {}
          else if (EAGAIN == errno || EWOULDBLOCK == errno)
          {
            err = AX_EAGAIN;
          }
        }
        else
        {
          if (wbytes == 0)
          {
            err = AX_SOCK_HUP;
          }
          else if (wbytes < len)
          {
            err = AX_EAGAIN;
          }
          //MLOG(INFO, "fd=%d write: %.*s wbytes=%ld err=%d", fd_, wbytes, buf, wbytes, err);
          write_done(wbytes);
        }
      }
    }
    return err;
  }

  int alloc_packet(Packet*& pkt) {
    return NULL == pkt_handler_? AX_NOT_INIT: pkt_handler_->alloc_packet(pkt);
  }
  int free_packet(Packet*& pkt) {
    return NULL == pkt_handler_? AX_NOT_INIT: pkt_handler_->free_packet(pkt);
  }
  int handle_packet(Packet* pkt) {
    return NULL == pkt_handler_? AX_NOT_INIT: pkt_handler_->handle_packet(id_, pkt);
  }
  int clone_packet(Packet* old_pkt, Packet*& new_pkt) {
    int err = AX_SUCCESS;
    if (NULL == old_pkt)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (AX_SUCCESS != (err = alloc_packet(new_pkt)))
    {}
    else
    {
      old_pkt->clone(new_pkt);
    }
    return err;
  }
  int send(Packet* pkt) {
    int err = AX_SUCCESS;
    Packet* new_pkt = NULL;
    if (NULL == pkt)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL == sched_)
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS != (err = clone_packet(pkt, new_pkt)))
    {}
    else if (AX_SUCCESS != (err = send_queue_.push((LinkNode*)new_pkt)))
    {}
    else
    {
      sched_->set_ready_flag(id_ | IOSched::WRITE_FLAG);
    }
    if (AX_SUCCESS != err)
    {
      if (NULL != new_pkt)
      {
        free_packet(new_pkt);
      }
    }
    return err;
  }
  int get_read_buf(char*& buf, int64_t& len) {
    int err = AX_SUCCESS;
    if (NULL != reading_pkt_)
    {}
    else if (AX_SUCCESS != (err = alloc_packet(reading_pkt_)))
    {}
    if (NULL != reading_pkt_)
    {
      err = reading_pkt_->get_buf(read_offset_, buf, len);
    }
    return err;
  }
  int read_done(int rbytes) {
    int err = AX_SUCCESS;
    rbytes_ += rbytes;
    if (0 == rbytes)
    {}
    else if (NULL == reading_pkt_)
    {
      err = AX_FATAL_ERR;
    }
    else if (!reading_pkt_->is_done(read_offset_ += rbytes))
    {}
    else if (!reading_pkt_->check_checksum())
    {
      err = AX_PKT_CHECKSUM_ERR;
    }
    else
    {
      if (AX_SUCCESS != (err = handle_packet(reading_pkt_)))
      {
        MLOG(WARN, "handle_packet() err=%d", err);
      }
      free_packet(reading_pkt_);
      reading_pkt_ = NULL;
      read_offset_ = 0;
    }
    return err;
  }
  int get_write_buf(char*& buf, int64_t& len) {
    int err = AX_SUCCESS;
    if (NULL != writing_pkt_)
    {}
    else if (AX_SUCCESS != (err = send_queue_.pop((LinkNode*&)writing_pkt_)))
    {}
    if (NULL != writing_pkt_)
    {
      err = writing_pkt_->get_buf(writen_offset_, buf, len);
      if (AX_SUCCESS != err)
      {
        ERR(err);
      }
    }
    return err;
  }
  int write_done(int wbytes) {
    int err = AX_SUCCESS;
    wbytes_ += wbytes;
    if (0 == wbytes)
    {}
    else if (NULL == writing_pkt_)
    {
      err = AX_FATAL_ERR;
    }
    else if (!writing_pkt_->is_done(writen_offset_ += wbytes))
    {}
    else
    {
      if (!writing_pkt_->check_checksum())
      {
        err = AX_PKT_CHECKSUM_ERR;
      }
      free_packet(writing_pkt_);
      writing_pkt_ = NULL;
      writen_offset_ = 0;
    }
    return err;
  }
  PacketHandler* pkt_handler_;
  LinkQueue send_queue_;
  int64_t writen_offset_;
  Packet* writing_pkt_;
  int64_t read_offset_;
  Packet* reading_pkt_;
};
  
class Nio
{
public:
  Nio() {}
  ~Nio(){}
public:
  int init(PacketHandler* handler, int port, int64_t capacity, char* buf) {
    int err = AX_SUCCESS;
    struct sockaddr_in addr;
    const char* ip = "127.0.0.1";
    MemChunkCutter cutter(calc_mem_usage(capacity), buf);
    if (NULL == handler || capacity <= 0 || NULL == buf)
    {
      err = AX_INVALID_ARGUMENT;
      MLOG(ERROR, "INVALID_ARGUMENT: handler=%p capacity=%ld buf=%p", handler, capacity, buf);
    }
    else if (AX_SUCCESS != (err = sched_.init(capacity, cutter.alloc(sched_.calc_mem_usage(capacity)))))
    {
      MLOG(ERROR, "io_sched.init()=>%d", err);
    }
    else if (AX_SUCCESS != (err = normal_sock_.init(&sched_, handler)))
    {
      MLOG(ERROR, "normal_sock.init()=>%d", err);
    }
    else if (port > 0 && AX_SUCCESS != (err = listen_sock_.init(&sched_, &normal_sock_, make_sockaddr(&addr, inet_addr(ip), port))))
    {
      MLOG(ERROR, "listen_sock.init()=>%d", err);
    }
    else if (AX_SUCCESS != (err = out_sock_cache_.init(&sched_, &normal_sock_, capacity, cutter.alloc(out_sock_cache_.calc_mem_usage(capacity)))))
    {
      MLOG(ERROR, "out_sock_cache.init()=>%d", err);
    }
    return err;
  }
  int64_t calc_mem_usage(int64_t capacity) {
    return sched_.calc_mem_usage(capacity) + out_sock_cache_.calc_mem_usage(capacity);
  }
  int sched(int64_t timeout) {
    return sched_.sched(timeout);
  }
  int send_packet(Id id, Packet* pkt) {
    int err = AX_SUCCESS;
    TcpSock* sock = NULL;
    if (AX_SUCCESS != (sched_.fetch(id, (Sock*&)sock)))
    {
      MLOG(ERROR, "fetch fail: maybe connection is del: id=%lx", id);
    }
    else if (AX_SUCCESS != (err = sock->send(pkt)))
    {
      MLOG(ERROR, "send fail: id=%lx", id);
    }
    if (NULL != sock)
    {
      sched_.revert(id);
      if (AX_SUCCESS != err)
      {
        sched_.del(id);
      }
    }
    return err;
  }
  int send_packet(Server server, Packet* pkt) {
    int err = AX_SUCCESS;
    TcpSock* sock = NULL;
    if (AX_SUCCESS != (err = out_sock_cache_.fetch(server, (Sock*&)sock)))
    {}
    else if (AX_SUCCESS != (err = sock->send(pkt)))
    {}
    if (NULL != sock)
    {
      out_sock_cache_.revert(sock, err);
    }
    return err;
  }
private:
  IOSched sched_;
  OutSockCache out_sock_cache_;
  ListenSock listen_sock_;
  TcpSock normal_sock_;
};

#endif /* __OB_AX_NIO_H__ */
