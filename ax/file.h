#ifndef __OB_AX_FILE_H__
#define __OB_AX_FILE_H__
#include "common.h"
#include "types.h"

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
  typedef SpinLock Lock;
  typedef SpinLock::Guard LockGuard;
public:
  AtomicFile(uint32_t magic, uint32_t bsize): magic_(magic), cur_block_(NULL), fd_(-1), bsize_(bsize), bcount_(0) {}
  ~AtomicFile() { destroy(); }
  int open(const char* path, char* buf, uint32_t& rbytes) {
    int err = AX_SUCCESS;
    struct stat _stat;
    LockGuard guard(lock_);
    if (NULL == path || NULL == buf)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if ((fd_ = open(path, O_CLOEXEC|O_RDWR|O_DIRECT)) < 0)
    {
      err = AX_IO_ERR;
    }
    else if (0 != flock(fd_, LOCK_EX|LOCK_NB))
    {
      err = AX_FLOCK_FAIL;
    }
    else if (0 != stat(path, &_stat))
    {
      err = OB_IO_ERROR;
    }
    else if ((bcount_ = _stat.st_size/bsize_) < 2)
    {
      err = AX_DATA_ERR;
    }
    else if (MAP_FAILED == (cur_block_ = mmap(NULL, bsize_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)))
    {
      block_ = NULL;
      err = AX_NO_MEM;
    }
    else
    {
      int64_t max_id = 0;
      for(uint32_t i = 0; i < bcount_; i++)
      {
        if (AX_SUCCESS != do_read_block(i, cur_block_))
        {}
        else if (cur_block_->id_ > max_id)
        {
          max_id = cur_block_->id_;
        }
      }
      if (max_id <= 0)
      {
        err = AX_NO_ENT;
        memset(cur_block_, 0, bsize_);
      }
      else if (AX_SUCCESS != (err = do_read_block(max_id, cur_block_)))
      {}
      else
      {
        memcpy(buf, cur_block_->buf_, cur_block_->len_);
        rbytes = cur_block_->len_;
      }
    }
    if (AX_SUCCESS != err)
    {
      destroy();
    }
    return err;
  }
  void destroy() {
    if (NULL != cur_block_)
    {
      munmap(cur_block_, bsize_);
      cur_block_ = NULL;
    }
    if (fd_ >= 0)
    {
      close(fd_);
      fd_ = -1;
    }
    fsize_ = 0;
  }
  int write(char* buf, int64_t len) {
    int err = AX_SUCCESS;
    LockGuard guard(lock_);
    if (NULL == buf || len <= 0 || sizeof(Block) + len > bsize_)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL == cur_block_)
    {
      err = AX_NOT_INIT;
    }
    else if (UINT64_MAX == cur_block_->seq_)
    {
      err = AX_SEQ_OVERFLOW;
    }
    else
    {
      cur_block_->set(magic_, len, buf, cur_block_->seq_+1);
      cur_block_->calc_checksum();
      int64_t total_size = cur_block_->get_record_len();
      if (total_size != unintr_pwrite(fd_, cur_block_, total_size, (cur_block_->id_ % bcount_) * bsize_))
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
  int do_read_block(Id id, Block* block) {
    int err = AX_SUCCESS;
    if (bsize_ != unintr_pread(fd_, block, bsize_, (id % bcount_) * bsize_))
    {
      err = AX_IO_ERR;
    }
    else if (!block->check_checksum(magic_))
    {
      err = AX_DATA_ERR;
    }
    return err;
  }
private:
  uint32_t magic_;
  Lock lock_;
  Block* cur_block_;
  int fd_;
  uint32_t bsize_;
  uint32_t bcount_;
};

#endif /* __OB_AX_FILE_H__ */
