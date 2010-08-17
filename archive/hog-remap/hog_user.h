#ifndef _HOG_USER_H_
#define _HOG_USER_H_

#include "hog.h"
#define _X86

#ifdef _X86
#define PAGE_SIZE (1<<12)
#else
#error "Not Support Arch!"
#endif

#define _hog_ __attribute__((section(".hog"), aligned(PAGE_SIZE)))
extern int _hog_fd_ ;
#define _str_(x) #x
#define _xstr_(x) _str_(x)
int hog_init();
int hog_destroy();
int hog_dump();
int hog_remap();
void* hog_alloc(unsigned long size);
void hog_free(void* addr, unsigned long len);
#endif /* _HOG_USER_H_ */
