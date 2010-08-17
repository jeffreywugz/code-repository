#include <linux/version.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/mm.h>
#include <asm/io.h>
#include "hog_pages.h"

#ifndef swap
#define swap(a, b) do { typeof(a) tmp = (a); (a) = (b); (b) = tmp;} while(0)
#endif

#define CACHE_COLOR_SIZE (1<<16)
struct hog_page {
	struct list_head list;
	struct page* pagp;
};

struct color_pages_t {
        int color;
        int n;
        struct list_head* hog_pages;
};

static void _pages_free(struct page** pages, int npages)
{
        int i;
        for(i = 0; i < npages; i++){
                if(pages[i])__free_page(pages[i]);
        }
        kfree(pages);
}

static struct page** _pages_new(int npages)
{
        struct page** pages = kzalloc(sizeof(struct page*) * npages,
                                      GFP_KERNEL);
        int i;
        if(!pages)goto out_err;
        for(i = 0; i < npages; i++){
                pages[i] = alloc_page(GFP_KERNEL);
                if(!pages[i])goto out_unalloc_pages;
        }
        return pages;
out_unalloc_pages:
        _pages_free(pages, npages);
out_err:        
        return NULL;
}


#if 0
struct page** hog_pages_new(int npages)
{
        struct page** pages = _hog_pages_new(2*npages);
        kmem_cache_t *hog_page_cache;
        int i;
        if(!pages)return pages;
        hog_page_cache = kmem_cache_create("hog_page", sizeof(struct hog_page),
                                         0, SLAB_HWCACHE_ALIGN, NULL, NULL);
        if (!hog_page_cache) {
                goto out_unalloc_pages;
        }
        for(i = npages; i < 2*npages; i++)
                __free_page(pages[i]);
out_unalloc_pages:
        hog_pages_free(pages, 2*npages);
        pages = 0;
out:
        kmem_cache_destroy(hog_page_cache);
        return pages;
}
#endif

static inline int page_color(struct page* pagp)
{
        return page_to_phys(pagp)%CACHE_COLOR_SIZE;
}

static inline int position(struct page* pagp)
{
        int color = page_color(pagp);
        return color <= CACHE_COLOR_SIZE/2 ? -1: +1;
}

struct page** _hog_pages_new(int npages)
{
        struct page** pages = _pages_new(2*npages);
        int i;
        int istart, iend;
        if(!pages)return pages;
        for(istart = 0, iend=2*npages-1; istart < iend;){
                while(position(pages[istart]) < 0)istart++;
                swap(pages[istart], pages[iend]);
                iend--;
                while(position(pages[iend]) >= 0)iend--;
                swap(pages[istart], pages[iend]);
                istart++;
        }
        printk(KERN_NOTICE "middle point: 0x%x\n", istart);
        for(i = npages; i < 2*npages; i++)
                __free_page(pages[i]);
        return pages;
}

struct pages_pool_t* hog_pages_new(int npages)
{
        struct pages_pool_t* pages_pool;
        struct page** pages;
        pages = _hog_pages_new(npages);
        if(!pages)goto out_err;
        pages_pool = kmalloc(sizeof(struct pages_pool_t), GFP_KERNEL);
        if(!pages_pool)goto out_unalloc_pages;
        pages_pool->n = npages;
        pages_pool->pages = pages;
        return pages_pool;
out_unalloc_pages:
        _pages_free(pages, npages);
out_err:        
        return NULL;
}

void hog_pages_del(struct pages_pool_t* pages_pool)
{
        _pages_free(pages_pool->pages, pages_pool->n);
        kfree(pages_pool);
}
