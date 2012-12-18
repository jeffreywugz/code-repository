#include <phi/kernel.h>
#include <phi/mm.h>
#include <phi/vm.h>
#include <string.h>

static struct vm_seg_t* vm_seg_new_()
{
        struct vm_seg_t* new;
        new=mem_alloc(sizeof(*new));
        return new;
}

struct vm_seg_t* vm_seg_new_at(u32 n, u32 addr)
{
        struct vm_seg_t* new;
        u32 pt;
        new=vm_seg_new_();
        new->pd=page_alloc(1);
        new->addr=addr;
        new->npt=n;
        pt=page_alloc(new->npt);
        pd_map_seg_usr(new->pd, pt, new->npt, 0, addr);
        return new;
}

struct vm_seg_t* vm_seg_new(u32 n)
{
        struct vm_seg_t* new;
        u32 addr;
        addr=page_alloc(n*PT_ENTRYS);
        new=vm_seg_new_at(n, addr);
        return new;
}

struct vm_seg_t* vm_seg_clone(struct vm_seg_t *old)
{
        struct vm_seg_t *new;
        u32 size=old->npt*PT_SIZE, addr;
        addr=page_alloc(old->npt*PT_ENTRYS);
        memcpy((void*)addr, (void*)virt_addr(old->addr), size);
        new=vm_seg_new_at(old->npt, phy_addr(addr));
        return new;
}
