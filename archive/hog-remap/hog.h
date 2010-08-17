#include <linux/ioctl.h>

struct hog_area_t {
        unsigned long start;
        unsigned long size;
};

#define HOG_IOC_MAGIC 'k'
#define HOG_IOC_DUMP _IO(HOG_IOC_MAGIC, 0)
#define HOG_IOC_REMAP _IOW(HOG_IOC_MAGIC, 1, struct hog_area_t)
#define HOG_IOC_MAXNR 2
