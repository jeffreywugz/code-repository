#include <phi/kernel.h>
#include <phi/mm.h>
#include <phi/vm.h>

static struct vma_t* vma_new_()
{
        struct vma_t *new;
        new=mem_alloc(sizeof(*new));
        return new;
}

struct vma_t* vma_new(u32 vaddr, struct vm_seg_t *seg)
{
        struct vma_t *new;
        new=vma_new_();
        new->vaddr=vaddr;
        new->seg=seg;
        return new;
}

struct vma_t* vma_new_at(u32 n, u32 vaddr, u32 addr)
{
        struct vm_seg_t *seg;
        struct vma_t *new;
        seg=vm_seg_new_at(n, addr);
        new=vma_new(vaddr, seg);
        return new;
}

struct vma_t* vma_clone(struct vma_t *old)
{
        struct vma_t *new;
        struct vm_seg_t *seg;
        seg=vm_seg_clone(old->seg);
        new=vma_new(old->vaddr, seg);
        return new;
}
