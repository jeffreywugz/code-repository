#include <phi/kernel.h>
#include <phi/task.h>
#include <phi/mm.h>

void page_fault_handler(struct state *st)
{
        u32 addr;
        __asm__ volatile ("movl %%cr2, %%eax":"=a"(addr));
        info("addr: %x", addr);
        halt();
}
