#ifdef MALLOC_MOD_ITEM_DEF
MALLOC_MOD_ITEM_DEF(AX_MOD_DEFAULT)
MALLOC_MOD_ITEM_DEF(AX_MOD_FIXED_ARRAY)
MALLOC_MOD_ITEM_DEF(AX_MALLOC_MOD_END)
#endif

#ifndef __OB_AX_MALLOC_H__
#define __OB_AX_MALLOC_H__

#include "ax_define.h"
#include "mcu.h"

enum
{
#define MALLOC_MOD_ITEM_DEF(name) name,
#include __FILE__
#undef MALLOC_MOD_ITEM_DEF
};
struct MallocModDesc
{
  MallocModDesc(): name_(NULL), used_(0), count_(0) {}
  ~MallocModDesc() {}
  void update(int64_t used) {
    FAA(&count_, used > 0? 1: -1);
    FAA(&used_, used);
  }
  int64_t to_string(char* buf, int64_t len) const;
  int64_t get_used() const { return used_; }
  int64_t get_count() const { return count_; }
  const char * name_;
  int64_t used_;
  int64_t count_;
};
    
class MallocModSet
{
public:
  enum {MOD_COUNT_LIMIT = AX_MALLOC_MOD_END };
  MallocModSet(): allocated_(0) {
#define MALLOC_MOD_ITEM_DEF(id) set_mod_name(ObModIds::id, #id);
#include __FILE__
#undef MALLOC_MOD_ITEM_DEF
  }
  ~MallocModSet() {}
  void update(int32_t mod_id, int64_t used) {
    mod_update(mod_id, used);
    FAA(&allocated_, used);
  }
  void print_mod_memory_usage();
private:
  int64_t mod_update(int32_t mod_id, int64_t used) { return mods_[(mod_id >= 0 && mod_id < MOD_COUNT_LIMIT)? mod_id: 0].update(used); }
  void set_mod_name(int32_t mod_id, const char* name) {
    if (mod_id >= 0 && mod_id < MOD_COUNT_LIMIT)
    {
      mods_[mod_id].name_ = name;
    }
  }
private:
  int64_t allocated_ CACHE_ALIGNED;
  MallocModDesc mods_[MOD_COUNT_LIMIT];
};

#endif /* __OB_AX_MALLOC_H__ */
