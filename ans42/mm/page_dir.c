#include <phi/kernel.h>
#include <phi/mm.h>
#include <string.h>

#define KERNEL_PAGE_ATTR 0x3
#define USR_PAGE_ATTR 0x7
extern char page_dir[PAGE_SIZE];

static inline void pt_entry_set(u32 p, u32 addr, u32 attr)
{
        *((u32*)p) = addr | attr; 
}

static inline void pd_entry_set(u32 p, u32 addr, u32 attr)
{
        *((u32*)p) = addr | attr; 
}

static inline void pt_map(u32 pt, u32 vaddr, u32 addr, u32 attr)
{
        pt_entry_set(pt+pt_offset(vaddr), addr, attr);
}

static inline void pt_map_n(u32 pt, u32 n, u32 vaddr, u32 addr, u32 attr)
{
        int i;
        for(i=0; i < n; i++, vaddr+=PAGE_SIZE, addr+=PAGE_SIZE)
                pt_map(pt, vaddr, addr, attr);
}

static inline void pt_map_whole(u32 pt, u32 addr, u32 attr)
{
        pt_map_n(pt, PT_ENTRYS, 0, addr, attr);
}

static inline void pd_map(u32 pd, u32 vaddr, u32 addr, u32 attr)
{
        pd_entry_set(pd+pd_offset(vaddr), addr, attr);
}

static inline void pd_map_n(u32 pd, u32 n, u32 vaddr, u32 addr, u32 attr)
{
        int i;
        for(i=0; i<n; i++, vaddr+=PT_SIZE, addr+=PAGE_SIZE)
                pd_map(pd, vaddr, addr, attr);
}

static void pt_map_seg(u32 pt, u32 n, u32 addr, u32 attr)
{
        int i;
        for(i = 0; i < n; i++)
                pt_map_whole(pt+PAGE_SIZE*i, addr+PT_SIZE*i, attr);
}

static void pd_map_seg(u32 pd, u32 pt, u32 n, u32 vaddr, u32 addr, u32 attr)
{
        pt_map_seg(pt, n, addr, attr);
        pd_map_n(pd, n, vaddr, phy_addr(pt), attr);
}

void pd_map_seg_kernel(u32 pd, u32 pt, u32 n, u32 vaddr, u32 addr)
{
        pd_map_seg(pd, pt, n, vaddr, addr, KERNEL_PAGE_ATTR);
}

void pd_map_seg_usr(u32 pd, u32 pt, u32 n, u32 vaddr, u32 addr)
{
        pd_map_seg(pd, pt, n, vaddr, addr, USR_PAGE_ATTR);
}

void pd_load(u32 pd)
{
        __asm__ volatile(
                "movl %%eax, %%cr3\n\t"
                "movl %%cr0, %%eax\n\t"
                "orl  $0x80000000, %%eax\n\t"
                "movl %%eax, %%cr0\n\t"
                :
                : "a"(phy_addr(pd)));
}

u32 pd_new()
{
        return page_alloc(1);
}

u32 pd_clone(u32 old)
{
        u32 new;
        new=pd_new();
        memcpy((void*)new, (void*)old, PAGE_SIZE);
        return new;
}
