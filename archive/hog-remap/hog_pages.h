
#ifndef _HOG_PAGES_H_
#define _HOG_PAGES_H_

struct pages_pool_t {
        struct page** pages;
        int n;
};
struct pages_pool_t* hog_pages_new(int npages);
void hog_pages_del(struct pages_pool_t* pages_pool);

#endif /* _HOG_PAGES_H_ */
