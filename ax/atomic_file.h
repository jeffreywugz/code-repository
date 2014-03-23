#ifndef __OB_AX_ATOMIC_FILE_H__
#define __OB_AX_ATOMIC_FILE_H__
#include "a0.h"
#include "lock.h"

inline int64_t unintr_pwrite(const int fd, const char *buf, const int64_t len, const int64_t offset)
{
  int wbytes = 0;
  int64_t writen_bytes = 0;
  for(writen_bytes = 0; writen_bytes < len; writen_bytes += wbytes)
  {
    if ((wbytes = ::pwrite(fd, (char*)buf + writen_bytes, len - writen_bytes, offset + writen_bytes)) >= 0)
    {}
    else if (errno == EINTR)
    {
      wbytes = 0;
    }
    else
    {
      break;
    }
  }
  return writen_bytes;
}

inline int64_t unintr_pread(const int fd, const char *buf, const int64_t len, const int64_t offset)
{
  int rbytes = 0;
  int64_t read_bytes = 0;
  for(read_bytes = 0; read_bytes < len; read_bytes += rbytes)
  {
    if ((rbytes = ::pread(fd, (char*)buf + read_bytes, len - read_bytes, offset + read_bytes)) >= 0)
    {}
    else if (errno == EINTR)
    {
      rbytes = 0;
    }
    else
    {
      break;
    }
  }
  return read_bytes;
}

class AtomicFile
{
public:
  typedef Record Block;
public:
  AtomicFile(): path_(NULL), magic_(0), cur_block_(NULL), fd_(-1), bsize_(0), fsize_(0) {}
  ~AtomicFile() {}
  static int create(const char* path, int bsize, int fsize, uint32_t magic) {
    int err = AX_SUCCESS;
    int fd = -1;
    if (NULL == path || bsize <= sizeof(Block) || (bsize & (bsize-1)) || fsize <= bsize || (fsize % bsize))
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if ((fd = open(path, O_CREAT|O_EXCL|O_WRONLY|O_DIRECT, S_IRUSR|S_IWUSR)) < 0)
    {
      err = AX_IO_ERR;
    }
    else if (0 != ftruncate(fd, fsize))
    {
      err = AX_IO_ERR;
    }
    else if (MAP_FAILED == (block = mmap(NULL, bsize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)))
    {
      block = NULL;
      err = AX_NO_MEM;
    }
    else
    {
      char init_block = 'i';
      block->set(magic, 1 /*len*/, &init_block, 1/*id*/);
      block->calc_checksum();
      int64_t total_size = block->get_record_len();
      if (total_size != uninter_pwrite(fd, block, total_size, (block->id_ * bsize) % fsize))
      {
        err = AX_IO_ERR;
      }
      munmap(block, bsize);
    }
    if (fd >= 0)
    {
      close(fd);
    }
    return err;
  }
  int open(const char* path, int bsize, uint32_t magic) {
    int err = AX_SUCCESS;
    struct stat _stat;
    RWLock::WriteGuard guard(lock_);
    bsize_ = bsize;
    if (NULL == path || bsize <= sizeof(Block) || (bsize & (bsize-1)))
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if ((fd_ = open(path, O_EXCL|O_RDWR|O_DIRECT)) < 0)
    {
      err = AX_IO_ERR;
    }
    else if (0 != stat(path, &_stat))
    {
      err = OB_IO_ERROR;
    }
    else if ((fsize_ = _stat.st_size) <= bsize || fsize_ % bsize)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (MAP_FAILED == (block_ = mmap(NULL, bsize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)))
    {
      block_ = NULL;
      err = AX_NO_MEM;
    }
    else
    {
      path_ = path;
      magic_ = magic;
      int64_t max_id = 0;
      for(int offset = 0; AX_SUCCESS == err && offset + bsize < fsize_; offset += bsize)
      {
        if (bsize != uninter_read(fd_, cur_block_, bsize, offset))
        {
          err = AX_IO_ERR;
        }
        else if (!cur_block_->check_checksum(magic))
        {
          err = AX_DATA_ERR;
        }
        else if ((cur_block_->id_ * bsize) % fsize_ != offset)
        {
          err = AX_FATAL_ERR;
        }
        else if (cur_block_->id_ > max_id)
        {
          max_id = cur_block_->id_;
        }
      }
      if (max_id <= 0)
      {
        err = AX_DATA_ERR;
      }
      else if (bsize != uninter_read(fd_, cur_block_, bsize, offset))
      {
        err = AX_IO_ERR;
      }
      else if (!cur_block_->check_checksum(magic))
      {
        err = AX_DATA_ERR;
      }
    }
    if (AX_SUCCESS != err)
    {
      destroy();
    }
    return err;
  }
  void destroy() {
    if (NULL != buf_)
    {
      munmap(buf_, bsize_);
      buf_ = NULL;
    }
    if (fd_ >= 0)
    {
      close(fd_);
      fd_ = -1;
    }
    path_ = NULL;
    magic_ = 0;
    bsize_ = 0;
    fsize_ = 0;
  }
  int read(char* buf, int64_t len, int64_t& data_len) {
    int err = AX_SUCCESS;
    int64_t data_len = 0;
    RWLock::ReadGuard guard(lock_);
    if (NULL == buf || len <= 0)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL == cur_block_)
    {
      err = AX_NOT_INIT;
    }
    else if ((data_len = cur_block_->get_payload_len()) > len)
    {
      err = AX_BUF_OVERFLOW;
    }
    else if (data_len <= 0)
    {
      err = AX_DATA_ERR;
    }
    else
    {
      memcpy(buf, cur_block_->buf_, data_len);
    }
    return err;
  }
  int write(char* buf, int64_t len) {
    int err = AX_SUCCESS;
    RWLock::WriteGuard guard(lock_);
    if (NULL == buf || len <= 0 || len > bsize_ - sizeof(Block))
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL == cur_block_)
    {
      err = AX_NOT_INIT;
    }
    else if (UINT64_MAX == cur_block->seq_)
    {
      err = AX_SEQ_OVERFLOW;
    }
    else
    {
      cur_block_->set(magic_, len, buf, cur_block_->seq_+1);
      cur_block_->calc_checksum();
      int64_t total_size = cur_block_->get_record_len();
      if (total_size != uninter_pwrite(fd_, cur_block_, total_size, (cur_block_->id_ * bsize_) % fsize_))
      {
        err = AX_IO_ERR;
      }
    }
    if (AX_SUCCESS != err)
    {
      destroy();
    }
    return err;
  }
private:
  const char* path_;
  uint32_t magic_;
  RWLock lock_;
  Block* cur_block_;
  int fd_;
  int bsize_;
  int fsize_;
};
#endif /* __OB_AX_ATOMIC_FILE_H__ */
