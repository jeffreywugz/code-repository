#include <phi/kernel.h>
#include <phi/arch.h>
#include <phi/mm.h>
#include <phi/task.h>
#include <phi/bitmap.h>
#include <string.h>

#define N_PID (1<<10)
static struct bitmap_t *tid_map;
extern char page_dir[PAGE_SIZE];
extern u32 _usr_data_start, _usr_data_end;
extern struct task_t *current;
extern struct tss sys_tss;

static inline int task_alloc_id()
{
        return bitmap_alloc(tid_map);
}

static inline void task_free_id(int id)
{
        bitmap_free(tid_map, id);
}

static inline struct task_t *task_new_()
{
        return mem_alloc(sizeof(struct task_t));
}

static inline struct task_t *task_clone_(struct task_t *old)
{
        struct task_t *new;
        new=task_new_();
        memcpy(new, old, sizeof(*new));
        return new;
}

static inline u32 task_kstack_new()
{
        return page_alloc(KSTACK_SIZE/PAGE_SIZE)+KSTACK_SIZE;
}

static inline u32 task_kstack_clone(u32 old)
{
        u32 new;
        new=task_kstack_new();
        memcpy((void*)(new-KSTACK_SIZE), (void*)(old-KSTACK_SIZE), KSTACK_SIZE);
        return new;
}
        
struct task_t *task_clone(struct task_t *old)
{
        struct task_t *new;
        new=task_clone_(old);
        new->id=task_alloc_id();
        new->kstack=task_kstack_clone(old->kstack);
        new->esp=new->kstack-old->kstack+old->esp;
        new->st=(struct state *)new->esp;
        new->vm=vm_clone(old->vm);
        return new;
}

int task_mutant(struct task_t *t)
{
        return 0;
}

int task_del(struct task_t *t)
{
        task_free_id(t->id);
        return 0;
}

void task_state_save(struct state *st)
{
        if(int_from_kernel(st))return;
        current->esp=st->_esp;
        current->st=st;
}

void task_state_restore(struct state *st)
{
        if(int_from_kernel(st))return;
        st->_esp=current->esp;
        sys_tss.esp0=current->kstack;
        vm_load(current->vm);
}

struct task_t *task_create_init()
{
        char kstack[KSTACK_SIZE];
        struct task_t *init;
        init=mem_alloc(sizeof(*init));
        init->kstack=(u32)kstack+KSTACK_SIZE;
        init->data_start=(u32)&_usr_data_start;
        init->data_end=(u32)&_usr_data_end;
        init->id=task_alloc_id();
        init->vm=vm_create_init();
        return init;
}

void task_init()
{
        tid_map=bitmap_new(N_PID);
}
