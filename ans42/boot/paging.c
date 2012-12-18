#include <phi/kernel.h>
#include <phi/mm.h>

#define PAGE_ATTR 0x3
extern char page_dir[PAGE_SIZE];

static inline void page_dir_boot_entry_set(u32 p, u32 addr)
{
        *((u32*)p) = addr | PAGE_ATTR; 
}

static inline void page_dir_boot_page_table_map(u32 pt, u32 addr)
{
        int i;
        for(i = 0; i < PT_ENTRYS; i++, addr+=PAGE_SIZE)
                page_dir_boot_entry_set(pt+4*i, addr);
}

static inline void page_dir_boot_entry_map(u32 pd, u32 vaddr, u32 pt)
{
        pd += vaddr/PT_SIZE*4;
        page_dir_boot_entry_set(pd, pt);
}

static void page_dir_boot_load(u32 pd)
{
        __asm__ volatile(
                "movl %%eax, %%cr3\n\t"
                "movl %%cr0, %%eax\n\t"
                "orl  $0x80000000, %%eax\n\t"
                "movl %%eax, %%cr0\n\t"
                :: "a"(phy_addr(pd)));
}

void paging_setup()
{
        u32 pd;
        static char pt0[PAGE_SIZE] __attribute__((aligned(PAGE_SIZE)));
        pd=phy_addr((u32)page_dir);
        page_dir_boot_page_table_map((u32)pt0, 0);
        page_dir_boot_entry_map(pd, 0, (u32)pt0);
        page_dir_boot_entry_map(pd, LOAD_OFFSET, (u32)pt0);
        page_dir_boot_load((u32)page_dir);
}
