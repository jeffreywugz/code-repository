#include <phi/kernel.h>
#include <phi/arch.h>
#include "mboot.h"\

void ksetup();
void gdt_init();
void idt_init();
void mm_init(u32 low, u32 high);
void dev_init();
void task_init();
void tqueue_init();
void usr_init();

static int mbt_get_high_mem(struct mboot_info *mbt, int low)
{
        struct mboot_mmap *mmap;
        for(mmap = (struct mboot_mmap*)mbt->mmap_addr;
            (u32)mmap < mbt->mmap_addr + mbt->mmap_length && mmap->base!=low;
            mmap = (struct mboot_mmap*)((u32)mmap + mmap->size + 4))
                ;
        return low+mmap->len;
}

void kmain(struct mboot_info *mbt, u32 magic)
{
        int low=MEM_BASE, high;
        high=mbt_get_high_mem(mbt, low);
        ksetup();
        gdt_init();
        idt_init();
        mm_init(low, high);
        dev_init();
        task_init();
        tqueue_init();
        usr_init();
}
