#include <phi/kernel.h>
#include <phi/mm.h>
#include <phi/task.h>

struct tss sys_tss;
struct task_t *current;
struct list_head *tqueue;

static void tss_init(u32 esp, u32 ss)
{
        sys_tss.esp0=esp;
        sys_tss.ss0=ss;
        __asm__ volatile ("ltr %%ax\n\t"::"a"(TSS));
}

void tqueue_print()
{
        struct task_t *task;
        list_for_each_entry(task, tqueue, head){
                info("id: %d\n", task->id);
        }
}

void tqueue_add(struct task_t *new)
{
        list_add(&new->head, tqueue);
}

void tqueue_del(struct task_t *del)
{
        list_del(&del->head);
}

void tqueue_sched(struct state *st)
{
        /* debug(tqueue_print()); */
        tqueue=tqueue->next;
        current=list_entry(tqueue, struct task_t, head);
}

void tqueue_init()
{
        struct task_t *init;
        init=task_create_init();
        tss_init(init->kstack, STACK_SEGMENT);
        tqueue=&init->head;
        INIT_LIST_HEAD(tqueue);
        current=init;
}

int sys_fork()
{
        struct task_t *new;
        new=task_clone(current);
        tqueue_add(new);
        current->st->eax=0;
        new->st->eax=new->id;
        return 0;
}
