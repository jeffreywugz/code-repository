
volatile int64_t __next_tid __attribute__((weak));
inline int64_t itid()
{
  static __thread int64_t tid = -1;
  return tid < 0? (tid = __sync_fetch_and_add(&__next_tid, 1)): tid;
}

struct TlValue
{
public:
  TlValue() { memset(items_, 0, sizeof(items_)); }
  ~TlValue() {}
  int64_t& get(){ return *(int64_t*)(items_ + itid()); }
private:
  char items_[AX_MAX_THREAD_NUM][CACHE_ALIGN_SIZE];
};
