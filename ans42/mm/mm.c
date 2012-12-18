#include <phi/kernel.h>
#include <phi/mm.h>

u32 page_init(u32 low, u32 high);
void mem_init(u32 low, u32 high);

void mm_init(u32 low, u32 high)
{
        low=page_init(low, high);
        high=virt_addr(LOAD_USR_PHY_ADDR);
        page_hold(high+PT_SIZE);
        mem_init(low, high);
}
