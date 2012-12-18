#include <phi/kernel.h>
#include <phi/list.h>
#include <phi/mm.h>

struct free_block
{
        struct list_head head;
        u32 len;
        char base[0];
};
static struct list_head *free_blocks;

/* static void mem_print() */
/* { */
/*         struct free_block *block; */
/*         list_for_each_entry(block, free_blocks, head){ */
/*                 info("[%x,%x] --> ", block->base, block->len); */
/*         } */
/*         info("[end]\n"); */
/* } */

static void merge_before(struct free_block *block)
{
        struct free_block *before;
        before=list_entry(block->head.prev, struct free_block, head);
        if((u32)before->base + before->len == (u32)block){
                before->len += block->len + sizeof(struct free_block);
                list_del(&block->head);
        }
}

static void merge_free_blocks(struct free_block *block)
{
        struct free_block *after;
        after=list_entry(block->head.next, struct free_block, head);
        merge_before(block);
        merge_before(after);
}

void* mem_alloc(u32 len)
{
        struct free_block *block, *alloc_block;
        list_for_each_entry(block, free_blocks, head){
                if(block->len > len)break;
        }
        if(&block->head == free_blocks)
                panic("no mem!");
        block->len -= len+sizeof(struct free_block);
        alloc_block = (struct free_block*)(block->base + block->len);
        alloc_block->len = len;
        return alloc_block->base;
}

void mem_free(void *p)
{
        struct free_block *block;
        list_for_each_entry(block, free_blocks, head){
                if((void*)block->base > p)break;
        }
        block=list_entry(block->head.prev, struct free_block, head);
        p=list_entry(p, struct free_block, base);
        list_add(&((struct free_block*)p)->head, &block->head);
        merge_free_blocks(p);
}

void mem_init(u32 low, u32 high)
{
        struct free_block *p;
        int len=high-low;
        p = (struct free_block*)low;
        free_blocks = &p->head;
        INIT_LIST_HEAD(free_blocks);
        p->len = 0;
        len -= sizeof(struct free_block)+4;
        p = (struct free_block*)(p->base + 4);
        p->len = len-sizeof(struct free_block);
        list_add(&p->head, free_blocks);
}
