#include <phi/kernel.h>
#include <phi/mm.h>
#include <phi/bitmap.h>
#include <string.h>

char page_dir[PAGE_SIZE]
__attribute__((section(".data.page_dir"),aligned(PAGE_SIZE)));
extern u32 _end;

struct page_t
{
        struct bitmap_t page_map;
        u32 low, high, pages;
};
struct page_t pages;

static inline u32 addr_from_page_no(u32 no)
{
        return virt_addr(PAGE_SIZE*no);
}
        
static inline u32 page_no_from_addr(u32 addr)
{
        return phy_addr(addr)/PAGE_SIZE;
}

u32 page_alloc(int c)
{
        return addr_from_page_no(bitmap_alloc_range(&pages.page_map, c));
}

void page_free(u32 p, int c)
{
        bitmap_free_range(&pages.page_map, page_no_from_addr(p), c);
}

void page_hold(u32 end)
{
        bitmap_hold(&pages.page_map, page_no_from_addr(end));
}

static u32 page_init_(u32 low, u32 high)
{
        pages.low=low;
        pages.high=high;
        pages.pages=high/PAGE_SIZE;
        return 0;
}

static u32 page_map_all(u32 pt)
{
        u32 start, end, npt;
        start=PT_SIZE;
        end=pages.high;
        npt=(end-start)/PT_SIZE;
        pd_map_seg_kernel((u32)page_dir, pt, npt, virt_addr(start), start);
        pt+=npt*PAGE_SIZE;
        return pt;
}

u32 page_init(u32 low, u32 high)
{
        u32 used, end;
        page_init_(low, high);
        end=(u32)&_end;
        used=bitmap_init(&pages.page_map, (void*)end, pages.pages);
        end=page_align_ceil(end+used);
        end=page_map_all(end);
        page_hold(end);
        return end;
}
