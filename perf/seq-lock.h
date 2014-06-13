class SeqLock
{
  struct Item
  {
    uint64_t value_;
  } CACHE_ALIGNED;
public:
  enum {N_THREAD=128};
  static const uint64_t N_THREAD_MASK = N_THREAD - 1;
  Item done_seq_[N_THREAD];
  uint64_t do_seq_ CACHE_ALIGNED;
  SeqLock(): do_seq_(0)
  {
    memset(done_seq_, 0, sizeof(done_seq_));
    __sync_synchronize();
  }
  ~SeqLock() {}
  int64_t lock()
  {
    uint64_t seq = __sync_add_and_fetch(&do_seq_, 1);
    volatile uint64_t& last_done_seq = done_seq_[(seq-1) & N_THREAD_MASK].value_;
    while(last_done_seq < seq - 1)
    {
      PAUSE() ;
    }
    __sync_synchronize();
    return seq;
  }
  int64_t unlock(int64_t seq)
  {
    volatile uint64_t& this_done_seq = done_seq_[seq & N_THREAD_MASK].value_;
    __sync_synchronize();
    this_done_seq = seq;
    return seq;
  }
};
