#include <phi/kernel.h>
#include <phi/mm.h>

u32 ustack_new()
{
        return page_alloc(STACK_SIZE/PAGE_SIZE)+STACK_SIZE;
}

u32 ustack_clone(u32 old_pd, u32 new_pd, u32 us)
{
        return 0;
}
