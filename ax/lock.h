#ifndef __OB_AX_LOCK_H__
#define __OB_AX_LOCK_H__

struct SpinLock
{
  struct Guard
  {
    Guard(SpinLock& lock): lock_(lock) {
      lock_.lock();
    }
    ~Guard() {
      lock_.unlock();
    }
    SpinLock& lock_;
  };

  SpinLock(): lock_(0) {}
  ~SpinLock() {}
  bool try_lock() {
    return CAS(&lock_, 0, 1);
  }

  bool lock() {
    while(!try_lock())
    {
      PAUSE();
    }
  }
  bool unlock() {
    return CAS(&lock_, 1, 0);
  }
  uint64_t lock_;
};

struct RWLock
{
  RWLock(): writer_id_(0), reader_ref_(0) {}
  ~RWLock() {}
  void rdlock() {
    while(!try_rdlock())
    {
      PAUSE();
    }
  }
  bool try_rdlock() {
    bool lock_succ = false;
    if (0 != AL(&writer_id_))
    {}
    else
    {
      FAA(&reader_ref_, 1);
      if (0 != AL(&writer_id_))
      {
        FAA(&reader_ref_, -1);
      }
      else
      {
        lock_succ = true;
      }
    }
    return lock_succ;
  }
  void rdunlock() {
    FAA(&reader_ref_, -1);
  }
  void wrlock() {
    while(!try_wrlock())
    {
      PAUSE();
    }
  }
  bool try_wrlock() {
    bool lock_succ = false;
    if (!CAS(&writer_id_, 0, 1))
    {}
    else
    {
      while(AL(&reader_ref_) != 0)
      {
        PAUSE();
      }
      lock_succ = true;
    }
    return lock_succ;
  }
  void wrunlock() {
    AS(&writer_id_, 0);
  }
  uint32_t writer_id_;
  uint32_t reader_ref_;
};

#endif /* __OB_AX_LOCK_H__ */

