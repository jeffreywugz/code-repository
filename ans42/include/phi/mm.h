#ifndef MM_H
#define MM_H

#include <phi/arch.h>

u32 pd_new();
u32 pd_clone(u32 old);
void pd_load(u32 pd);
void pd_map_seg_kernel(u32 pd, u32 pt, u32 n, u32 vaddr, u32 addr);
void pd_map_seg_usr(u32 pd, u32 pt, u32 n, u32 vaddr, u32 addr);

u32 page_alloc(int c);
void page_free(u32 p, int c);
void page_hold(u32 end);
void* mem_alloc(u32 len);
void mem_free(void* p);

static inline u32 align_floor(u32 addr, u32 align)
{
        return addr & ~(align-1);
}

static inline u32 align_ceil(u32 addr, u32 align)
{
        return align_floor(addr+align-1, align);
}

static inline u32 phy_addr(u32 addr)
{
        return addr-LOAD_OFFSET;
}

static inline u32 virt_addr(u32 addr)
{
        return addr+LOAD_OFFSET;
}

static inline u32 page_align_floor(u32 addr)
{
        return align_floor(addr, PAGE_SIZE);
}

static inline u32 page_align_ceil(u32 addr)
{
        return align_ceil(addr, PAGE_SIZE);
}

static inline bool aligned(u32 addr, u32 align)
{
        return !(addr&(align-1));
}

static inline bool page_aligned(u32 addr)
{
        return aligned(addr, PAGE_SIZE);
}

static inline bool vma_aligned(u32 addr)
{
        return aligned(addr, PT_SIZE);
}
#endif //MM_H
