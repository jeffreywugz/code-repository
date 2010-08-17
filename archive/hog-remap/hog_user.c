#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include "hog_user.h"
#include "config.h"

int _hog_fd_ = -1;

#if SUPPORT_KERNEL_MODULE
int hog_init()
{
        _hog_fd_ = open(_xstr_(HOG_DEVICE), O_RDWR|O_SYNC);
        return _hog_fd_;
}

int hog_destroy()
{
        return close(_hog_fd_);
}

int hog_dump()
{
        return ioctl(_hog_fd_, HOG_IOC_DUMP);
}

int hog_remap(unsigned long start, unsigned long size)
{
        struct hog_area_t area = {start, size};
        return ioctl(_hog_fd_, HOG_IOC_REMAP, &area);
}
#else
int hog_init()
{
        return 1;
}

int hog_destroy()
{
        return 0;
}

int hog_dump()
{
        return 0;
}

int hog_remap()
{
        return 0;
}
#endif

static unsigned long page_align(unsigned long x)
{
        return (x + getpagesize() - 1) & (~(getpagesize()-1));
}

#if SUPPORT_KERNEL_MODULE && SUPPORT_DYNAMIC_REMAP
void* hog_alloc(unsigned long size)
{
        int fd;
        void *addr;
        int len = page_align(size);

        if ((fd=open("hog.node", O_RDWR|O_SYNC))<0){
                perror("open");
                return NULL;
        }

        addr = mmap(0, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  
        if (addr == MAP_FAILED){
                perror("mmap");
                return NULL;
        }

        close(fd);
        return addr;
}

void hog_free(void* addr, unsigned long len)
{
        munmap(addr, len);
}
#else
void* hog_alloc(unsigned long size)
{
        return malloc(size);
}

void hog_free(void* addr, unsigned long len)
{
        free(addr);
}
#endif

#if SUPPORT_STATIC_REMAP
extern char _hog_start_, _hog_end_;
#endif

__attribute__((constructor)) void _hog_init()
{
        int err = 0;
        printf("_hog_init:\n");
        if(hog_init() <= 0){
                perror("hog_init");
                exit(-1);
        }
#if SUPPORT_STATIC_REMAP
        printf("_hog_start_=%p, _hog_end_=%p\n", &_hog_start_, &_hog_end_);
        err = hog_remap((unsigned long)&_hog_start_,
                        (unsigned long)(&_hog_end_ - &_hog_start_));
#endif        
        if(err < 0)perror("hog_remap:");
        
}

__attribute__((destructor)) void _hog_destroy()
{
        printf("_hog_destroy:\n");
        assert(hog_destroy() == 0);
}
