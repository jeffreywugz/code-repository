#include <phi/kernel.h>
#include <phi/arch.h>
#include <phi/task.h>

void irq_send_eoi();
typedef void (*int_handler_t)(struct state* st);
typedef int (*sys_call_handler_t)(int a, int b, int c);
extern void *sys_call_table[0];

static const char *exception_messages[32] =
{
#include "exception_message.h"
};

extern void page_fault_handler(struct state *st);
extern void timer_handler(struct state *st);
extern void kbd_handler(struct state *st);

static void* exception_routines[32]={
        0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, page_fault_handler, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 
};

static void* irq_routines[16]={
        timer_handler, kbd_handler, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
};

static void int_print(struct state *st)
{
        const char *from_where[]={"user", "kernel"};
        const char *where=from_where[int_from_kernel(st)];
        const char *msg="maybe irq";
        int no=st->int_no, err=0xff&(st->err_code), eip=st->eip;
        if(no<32)msg=exception_messages[no];
        else if(no==0x80)msg="sys call";
        info("int from %s: eip=%x no=%x error=%x [%s]\n",
               where, eip, no, err, msg);
}

static void err_handler(struct state *st)
{
        info("handler missing...\n");
        halt();
}

static inline void* _get_handler(void* h)
{
        if(h)return h;
        return err_handler;
}

static void exception_handler(struct state *st)
{
        int_handler_t handler;
        int no=st->int_no;
        _dbg(int_print(st));
        handler = _get_handler(exception_routines[no]);
        handler(st);
}

static void sys_call_handler(struct state *st)
{
        sys_call_handler_t handler;
        handler=_get_handler(sys_call_table[st->eax]);
        handler(st->ebx, st->ecx, st->edx);
}

static void irq_handler(struct state *st)
{
        int_handler_t handler;
        int no=st->int_no;
        handler = _get_handler(irq_routines[no-0x20]);
        handler(st);
        irq_send_eoi(no);
}

static void* int_get_handler(struct state *st)
{
        int no=st->int_no;
        if(no<0x20){
                return exception_handler;
        } else if(no<0x30){
                return irq_handler;
        } else if(no==0x80){
                return sys_call_handler;
        } else {
                return err_handler;
        }
}

void int_handler(struct state *st)
{
        int_handler_t handler;
        task_state_save(st);
        handler=int_get_handler(st);
        handler(st);
        task_state_restore(st);
}
