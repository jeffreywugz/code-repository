#include <phi/kernel.h>
#include <phi/arch.h>

void irq_init();

#define N_IDT_ENTRY 256

struct idt_entry
{
        u32 base;
        u16 sel;
        u8 flags;
} __attribute__((packed));

struct enc_idt_entry
{
        u16 base_lo;
        u16 sel;
        u8 always0;
        u8 flags;
        u16 base_hi;
} __attribute__((packed));

struct idt_ptr
{
        unsigned short limit;
        unsigned int base;
} __attribute__((packed));

extern struct idt_entry idt[N_IDT_ENTRY];
struct enc_idt_entry enc_idt[N_IDT_ENTRY]
__attribute__((section(".data.idt"), aligned(PAGE_SIZE)));

static void idt_encode(struct enc_idt_entry *dest, struct idt_entry *src)
{
        /* The interrupt routine's base address */
        dest->base_lo = (src->base & 0xFFFF);
        dest->base_hi = (src->base >> 16) & 0xFFFF;

        /* The segment or 'selector' that this IDT entry will use
         *  is set here, along with any access flags */
        dest->sel = src->sel;
        dest->always0 = 0;
        dest->flags = src->flags;
}

static void idt_load()
{
        struct idt_ptr idtp={.limit= sizeof(enc_idt)-1, .base=(u32)enc_idt};
        __asm__ volatile ("lidt %0\n\t"::"m" (idtp));
}

static void idt_set()
{
        int i;
        for(i=0; i<N_IDT_ENTRY; i++)
                idt_encode(enc_idt+i, idt+i);
}

void idt_init()
{
        irq_init();
        idt_set();
        idt_load();
}
