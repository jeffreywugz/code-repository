#ifndef VM_H
#define VM_H

struct vm_t
{
        u32 pd;
        struct list_head vma;
};

struct vma_t
{
        struct list_head head;
        u32 vaddr;
        struct vm_seg_t *seg;
};

struct vm_seg_t
{
        u32 type;
        u32 addr;
        u32 pd;
        u32 npt;
};

struct vm_t* vm_create_init();
struct vm_t* vm_clone(struct vm_t *new);
void vm_copy(struct vm_t *vm, u32 vstart, u32 vend);
void vm_load(struct vm_t *new);
struct vm_seg_t* vm_seg_new(u32 n);
struct vm_seg_t* vm_seg_new_at(u32 n, u32 addr);
struct vm_seg_t* vm_seg_clone(struct vm_seg_t *old);
struct vma_t* vma_new(u32 vaddr, struct vm_seg_t *seg);
struct vma_t* vma_new_at(u32 n, u32 vaddr, u32 addr);
struct vma_t* vma_clone(struct vma_t *old);


#endif //VM_H
