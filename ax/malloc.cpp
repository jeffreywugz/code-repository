#include "malloc.h"

MemAllocator& get_global_mem_allocator()
{
  static MemAllocator allocator;
  return allocator;
}

void* ax_malloc(size_t size, int mod_id)
{
  get_global_mem_allocator().alloc(size, mod_id);
}

void ax_free(void* p)
{
  get_global_mem_allocator().free(p);
}
