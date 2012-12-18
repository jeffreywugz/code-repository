#include <phi/mm.h>
#include <stdint.h>
#include <string.h>

struct bitmap_t
{
        char *base;
        u32 len;
};

static inline int bitmap_buffer_size(int len)
{
        return len;
}

static inline int bitmap_init(struct bitmap_t *p, void* base, int len)
{
        p->base=base;
        p->len=len;
        memset(p->base, 0, p->len);
        return len;
}

static inline struct bitmap_t* bitmap_new(int len)
{
        char *buf=mem_alloc(bitmap_buffer_size(len));
        struct bitmap_t *p=mem_alloc(sizeof(*p));
        bitmap_init(p, buf, len);
        return p;
}

static inline int bitmap_check(char *p, int c)
{
        int i;
        for(i=0; i<c; i++)
                if(p[i])return 0;
        return 1;
}

static inline void bitmap_occupy_range(struct bitmap_t *p, int i, int c)
{
        memset(p->base+i, 1, c);
}

static inline void bitmap_occupy(struct bitmap_t *p, int i)
{
        bitmap_occupy_range(p, i, 1);
}

static inline void bitmap_hold(struct bitmap_t *p, int i)
{
        bitmap_occupy_range(p, 0, i);
}

static inline int bitmap_alloc_range(struct bitmap_t *p, int c)
{
        int i;
        for(i=0; i < p->len-c+1; i++)
                if(bitmap_check(p->base+i, c))break;
        if(i==p->len-c+1)return -1;
        bitmap_occupy_range(p, i, c);
        return i;
}

static inline void bitmap_free_range(struct bitmap_t *p, int i, int c)
{
        memset(p->base+i, 0, c);
}

static inline int bitmap_alloc(struct bitmap_t *p)
{
        return bitmap_alloc_range(p, 1);
}

static inline void bitmap_free(struct bitmap_t *p, int i)
{
        bitmap_free_range(p, i, 1);
}
