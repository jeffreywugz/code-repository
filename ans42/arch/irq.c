#include <phi/kernel.h>

/* remap irq to 16-47 */
static void irq_remap()
{
        outb(0x11, 0x20);
        outb(0x11, 0xA0);
        outb(0x20, 0x21);
        outb(0x28, 0xA1);
        outb(0x04, 0x21);
        outb(0x02, 0xA1);
        outb(0x01, 0x21);
        outb(0x01, 0xA1);
        outb(0x0, 0x21);
        outb(0x0, 0xA1);
}

void irq_send_eoi(int no)
{
        if (no >= 40){
                outb(0x20, 0xA0);
        }
        outb(0x20, 0x20);
}

void irq_init()
{
        irq_remap();
}
