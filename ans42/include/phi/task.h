#ifndef TASK_H
#define TASK_H

#include <phi/list.h>
#include <phi/vm.h>
#include <phi/arch.h>

enum task_error {
        SUCCESS=0, EGENERAL, 
};

struct task_t
{
        u32 esp;    //actual position of esp
        u32 ss;     //actual stack segment.
        u32 kstack; //stacktop of kernel stack
        u32 ustack; //stacktop of user stack
        struct vm_t *vm;
        struct state *st;
        u32 ustart, uend;
        u32 data_start, data_end; 
        u32 id;
        u32 entry;
        struct list_head head;
};

void task_state_save(struct state *st);
void task_state_restore(struct state *st);
struct task_t *task_create_init();
struct task_t *task_clone(struct task_t *old);


#endif //TASK_H
