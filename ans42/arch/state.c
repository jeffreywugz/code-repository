#include <phi/kernel.h>
#include <phi/arch.h>
#include <phi/mm.h>

struct state* state_new()
{
        return mem_alloc(sizeof(struct state));
}

void state_save(struct state *st)
{
        
}

void state_restore(struct state *st)
{
        
}
