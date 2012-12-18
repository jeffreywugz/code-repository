#ifndef ARCH_H
#define ARCH_H

#define PAGE_SIZE 0x1000
#define PT_SIZE 0x400000
#define PT_ENTRYS 1024
#define MEM_BASE 0x100000

static inline u32 page_mask(u32 addr)
{
        return addr&0xfff;
}

static inline u32 pt_mask(u32 addr)
{
        return addr&0x3fffff;
}

static inline u32 pt_offset(u32 addr)
{
        return addr/PAGE_SIZE*4;
}

static inline u32 pd_offset(u32 addr)
{
        return addr/PT_SIZE*4;
}

struct state
{
        u32 _esp;
        u32 gs, fs, es, ds;
        u32 edi, esi, ebp, esp, ebx, edx, ecx, eax;
        u32 int_no, err_code;
        u32 eip, cs, eflags, user_esp, user_ss;    
} __attribute__((packed));

struct tss
{
        u16 backlink, __blh;
        u32 esp0;
        u16 ss0, __ss0h;
        u32 esp1;
        u16 ss1, __ss1h;
        u32 esp2;
        u16 ss2, __ss2h;
        u32 cr3;
        u32 eip;
        u32 eflags;
        u32 eax, ecx, edx, ebx;
        u32 esp, ebp, esi, edi;
        u16 es, __esh;
        u16 cs, __csh;
        u16 ss, __ssh;
        u16 ds, __dsh;
        u16 fs, __fsh;
        u16 gs, __gsh;
        u16 ldt, __ldth;
        u16 trace, bitmap;
} __attribute__((packed));

static inline bool int_from_kernel(struct state *st)
{
        return (st->cs&3)==0;
}

#endif //ARCH_H
