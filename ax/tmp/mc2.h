#ifndef __OB_AX_MC2_H__
#define __OB_AX_MC2_H__

struct TlValue
{
public:
  TlValue() { memset(items_, 0, sizeof(items_)); }
  ~TlValue() {}
  int64_t& get(){ return *(int64_t*)(items_ + itid()); }
private:
  char items_[AX_MAX_THREAD_NUM][CACHE_ALIGN_SIZE];
};
#endif /* __OB_AX_MC2_H__ */
