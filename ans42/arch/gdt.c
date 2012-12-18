#include <phi/kernel.h>
#include <phi/arch.h>

extern struct tss sys_tss;

struct gdt_entry
{
        u32 base;
        u32 limit;
        u8 access;
        u8 granularity;
};

struct enc_gdt_entry
{
        u16 limit_low;
        u16 base_low;
        u8 base_middle;
        u8 access;
        u8 granularity;
        u8 base_high;
} __attribute__((packed));

struct gdt_ptr {
        u16 limit;
        u32 base;
} __attribute__((packed));

static struct gdt_entry gdt[]={
        {.base=0, .limit=0, .access=0, .granularity=0}, // cannot be used
        {.base=0, .limit=0xffffffff, .access=0x9A, .granularity=0xCF}, // code
        {.base=0, .limit=0xffffffff, .access=0x92, .granularity=0xCF}, // data
        {.base=0, .limit=0xffffffff, .access=0xFA, .granularity=0xCF}, // code
        {.base=0, .limit=0xffffffff, .access=0xF2, .granularity=0xCF}, // data
        {.base=(u32)(&sys_tss), .limit=sizeof(sys_tss)-1, .access=0x89, .granularity=0xCF}, //tss
};
static struct enc_gdt_entry enc_gdt[sizeof(gdt)/sizeof(struct gdt_entry)]
__attribute__((section(".data.gdt"), aligned(32)));

static void gdt_encode(struct enc_gdt_entry *dest, struct gdt_entry *src) 
{
        /* Setup the descriptor base address */
        dest->base_low = (src->base & 0xFFFF);
        dest->base_middle = (src->base >> 16) & 0xFF;
        dest->base_high = (src->base >> 24) & 0xFF;

        /* Setup the descriptor limits */
        dest->limit_low = (src->limit & 0xFFFF);
        dest->granularity = ((src->limit >> 16) & 0x0F);

        /* Finally, set up the granularity and access flags */
        dest->granularity |= (src->granularity & 0xF0);
        dest->access = src->access;
}

static void gdt_load()
{
        struct gdt_ptr gdtp={.limit=sizeof(enc_gdt)-1, .base=(u32)enc_gdt};
        __asm__ volatile (
                "lgdt %0\n\t"
                "ljmp %1, $reload_segments\n\t"
                "reload_segments:\n\t"
                "movw %2, %%ax\n\t"
                "movw %%ax, %%ds\n\t"
                "movw %%ax, %%es\n\t"
                "movw %%ax, %%fs\n\t"
                "movw %%ax, %%gs\n\t"
                "movw %%ax, %%ss\n\t"
                :: "m" (gdtp),"g"(CODE_SEGMENT),"g"(DATA_SEGMENT):"ax");
}

static void gdt_set()
{
        int i;
        for(i=0; i<sizeof(gdt)/sizeof(struct gdt_entry); i++)
                gdt_encode(enc_gdt+i, gdt+i);
}

void gdt_init()
{
        gdt_set();
        gdt_load();
}
