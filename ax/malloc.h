#ifdef MALLOC_MOD_ITEM_DEF
MALLOC_MOD_ITEM_DEF(AX_MOD_DEFAULT)
MALLOC_MOD_ITEM_DEF(AX_MOD_FIXED_ARRAY)
MALLOC_MOD_ITEM_DEF(AX_MALLOC_MOD_END)
#endif

#ifndef __OB_AX_MALLOC_H__
#define __OB_AX_MALLOC_H__

#include "a0.h"

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
#define MALLOC_MOD_ITEM_DEF(id) set_mod_name(id, #id);
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
  void mod_update(int32_t mod_id, int64_t used) { return mods_[(mod_id >= 0 && mod_id < MOD_COUNT_LIMIT)? mod_id: 0].update(used); }
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

class MemAllocator
{
public:
  struct Block
  {
    Block(int mod_id, int64_t size): mod_id_(mod_id), size_(size) {}
    ~Block() {}
    int64_t get_alloc_size() const { return size_ + sizeof(*this); }
    int mod_id_;
    int64_t size_;
    char buf_[0];
  };
public:
  void* alloc(int64_t size, int mod_id) {
    Block* block = alloc_block(size, mod_id);
    return NULL != block? block->buf_: NULL;
  }
  void free(void* p) {
    if (NULL != p)
    {
      Block* block = (Block*)(p) - 1;
      free_block(block);
    }
  }
protected:
  Block* alloc_block(int64_t size, int mod_id) {
    Block* block = NULL;
    if (size > 0)
    {
      int64_t alloc_size = size + sizeof(Block);
      block = (Block*)::malloc(alloc_size);
      if (NULL != block)
      {
        mod_set_.update(mod_id, size);
        new(block) Block(mod_id, size);
      }
    }
    return block;
  }
  void free_block(Block* block)
  {
    if (NULL != block)
    {
      mod_set_.update(block->mod_id_, -block->get_alloc_size());
      ::free(block);
    }
  }
private:
  MallocModSet mod_set_;
};

inline MemAllocator& get_global_mem_allocator()
{
  static MemAllocator allocator;
  return allocator;
}

inline void* ax_malloc(size_t size, int mod_id)
{
  return get_global_mem_allocator().alloc(size, mod_id);
}

inline void ax_free(void* p)
{
  get_global_mem_allocator().free(p);
}

#endif /* __OB_AX_MALLOC_H__ */
