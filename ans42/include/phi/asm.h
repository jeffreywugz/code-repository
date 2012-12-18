#ifndef ASM_H
#define ASM_H

#define sti() __asm__ ("sti"::)
#define cli() __asm__ ("cli"::)
#define nop() __asm__ ("nop"::)
#define halt() __asm__("hlt"::)
#define iret() __asm__("iret"::)
#define outb(value,port) \
__asm__ volatile ("outb %%al,%%dx"::"a" (value),"d" (port))
#define inb(port) ({ \
unsigned char _v; \
__asm__ volatile ("inb %%dx,%%al":"=a" (_v):"d" (port)); \
_v; \
})

#define stack_top() ({ \
u32 _v; \
__asm__ volatile ("movl %%esp, %%eax":"=a" (_v):);  \
_v; \
})

#define switch_to_user_mode() \
__asm__ volatile ("movl %%esp,%%eax\n\t" \
         "pushl %1\n\t" \
         "pushl %%eax\n\t" \
         "pushfl\n\t" \
         "pushl %0\n\t" \
         "pushl $1f\n\t" \
         "iret\n" \
         "1:\t movw %1,%%ax\n\t" \
         "movw %%ax,%%ds\n\t" \
         "movw %%ax,%%es\n\t" \
         "movw %%ax,%%fs\n\t" \
         "movw %%ax,%%gs" \
         ::"g"(USER_CODE_SEGMENT),"g"(USER_DATA_SEGMENT):"eax")

#endif //ASM_H
