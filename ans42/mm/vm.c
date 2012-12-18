#include <phi/kernel.h>
#include <phi/mm.h>
#include <phi/vm.h>
#include <phi/bitmap.h>
#include <string.h>

extern char page_dir[PAGE_SIZE];

void vm_print(struct vm_t *vm)
{
        struct vma_t *vma;
        list_for_each_entry(vma, &vm->vma, head){
                info("[%x=>%x:%x]-->",
                     vma->vaddr, vma->seg->addr, vma->seg->npt);
        }
        info("[end]\n");
}

static inline void vm_map_seg(struct vm_t *vm, u32 vaddr, struct vm_seg_t *seg)
{
        memcpy((void*)(vm->pd+pd_offset(vaddr)), (void*)(seg->pd), seg->npt*4);
}

static inline void vm_add_vma(struct vm_t *vm, struct vma_t *vma)
{
        struct vma_t *p;
        list_for_each_entry(p, &vm->vma, head){
                if(vma->vaddr < p->vaddr)break;
        }
        list_add_tail(&vma->head, &p->head);
}

static inline void vm_map_vma(struct vm_t *vm, struct vma_t *vma)
{
        vm_map_seg(vm, vma->vaddr, vma->seg);
        vm_add_vma(vm, vma);
}

static struct vm_t* vm_new_()
{
        struct vm_t *new;
        new=mem_alloc(sizeof(struct vm_t));
        INIT_LIST_HEAD(&new->vma);
        return new;
}

struct vm_t* vm_new()
{
        struct vm_t *new;
        new=vm_new_();
        new->pd=pd_new();
        return new;
}

void vm_unmap(struct vm_t *vm, u32 vstart, u32 vend)
{
        /* TODO: add code here */
}

struct vm_t* vm_create_init_()
{
        struct vm_t *vm;
        vm=vm_new_();
        vm->pd=(u32)page_dir;
        return vm;
}

struct vm_t* vm_create_init()
{
        struct vm_t *vm;
        struct vma_t *vma;
        vm=vm_create_init_();
        vma=vma_new_at(1, 0, LOAD_USR_PHY_ADDR);
        vm_map_vma(vm, vma);
        _dbg(vm_print(vm));
        return vm;
}

struct vm_t* vm_clone(struct vm_t *old)
{
        struct vm_t *new;
        struct vma_t *vma, *new_vma;
        new=vm_new_();
        new->pd=pd_clone(old->pd);
        list_for_each_entry(vma, &old->vma, head){
                new_vma=vma_clone(vma);
                vm_map_vma(new, new_vma);
        }
        _dbg(vm_print(new));
        return new;
}

void vm_load(struct vm_t *new)
{
        pd_load(new->pd);
}
