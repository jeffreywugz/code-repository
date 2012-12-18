#include <phi/kernel.h>
#include <phi/arch.h>
#define __LIBRARY__
#include <stdio.h>
#include <unistd.h>

_syscall1(int,write, char *, s);
_syscall0(int,fork);

int usr_printf(const char *fmt, ...);
char usr_stack[STACK_SIZE] __attribute__((aligned(PAGE_SIZE)));
int errno;

void usr_init()
{
        int i, id;
        __asm__ volatile (
                "sti\n\t"
                "pushl %2\n\t"
                "pushl %3\n\t"
                "pushfl\n\t"
                "pushl %0\n\t"
                "pushl $usr_start\n\t"
                "iret\n\t"
                "usr_start:\n\t"
                "movw %1, %%ax\n\t"
                "movw %%ax, %%ds\n\t"
                "movw %%ax, %%es\n\t"
                "movw %%ax, %%fs\n\t"
                "movw %%ax, %%gs\n\t"
                ::
                 "g"(USER_CODE_SEGMENT),
                 "g"(USER_DATA_SEGMENT),
                 "g"(USER_STACK_SEGMENT),
                 "g"((u32)&usr_stack+STACK_SIZE):"eax");
        for(i=0; i<4; i++){
                id=fork();
                if(id)break;
        }
        usr_printf("hello from process: %d\n", id);
        while(1);
}
