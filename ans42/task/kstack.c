#include <phi/kernel.h>
#include <phi/mm.h>
#include <string.h>

u32 kstack_new()
{
        return page_alloc(KSTACK_SIZE/PAGE_SIZE)+KSTACK_SIZE;
}

u32 kstack_clone(u32 old)
{
        u32 new;
        new=kstack_new();
        memcpy((void*)(new-KSTACK_SIZE), (void*)(old-KSTACK_SIZE), KSTACK_SIZE);
        return new;
}
