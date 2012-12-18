#include <phi/kernel.h>
#include <phi/arch.h>

int ticks = 0;
void tqueue_sched(struct state *st);

void timer_init(int hz)
{
        int divisor = 1193180 / hz;       /* Calculate our divisor */
        outb(0x36, 0x43);             /* Set our command byte 0x36 */
        outb(divisor & 0xFF, 0x40);   /* Set low byte of divisor */
        outb(divisor >> 8, 0x40);     /* Set high byte of divisor */
}

void timer_handler(struct state *st)
{
        ticks++;
        if(int_from_kernel(st))
                return;
        tqueue_sched(st);
}
