#include <linux/version.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/mm.h>
#include <linux/sched.h>
#include <asm/io.h>
#include <asm/uaccess.h>
#include "hog.h"
#include "hog_pages.h"

MODULE_LICENSE("Dual BSD/GPL");

static dev_t hog_dev;
static struct cdev hog_cdev;

static int hog_open(struct inode *inode, struct file *filp);
static int hog_release(struct inode *inode, struct file *filp);
static int hog_mmap(struct file *filp, struct vm_area_struct *vma);
static int hog_ioctl(struct inode *, struct file *,
                     unsigned int cmd, unsigned long arg);
static struct file_operations hog_fops = {
        .open = hog_open,
        .release = hog_release,
        .ioctl = hog_ioctl,
        .mmap = hog_mmap,
        .owner = THIS_MODULE,
};

static int hog_open(struct inode *inode, struct file *filp)
{
        return 0;
}

static int hog_release(struct inode *inode, struct file *filp)
{
        return 0;
}

static void current_dump_vma(void)
{
        struct mm_struct *mm = current->mm;
        struct vm_area_struct *vma;
        printk("The current process is %s\n",current->comm);
        printk("dump_vma_list\n");
        down_read(&mm->mmap_sem);
        for (vma = mm->mmap;vma; vma = vma->vm_next) {
                printk("VMA 0x%lx-0x%lx ",
                       vma->vm_start, vma->vm_end);
                if (vma->vm_flags & VM_WRITE)
                        printk("WRITE ");
                if (vma->vm_flags & VM_READ)
                        printk("READ ");
                if (vma->vm_flags & VM_EXEC)
                        printk("EXEC ");
                printk("\n");
        }
        up_read(&mm->mmap_sem);
}

static struct vm_area_struct* current_find_vma(unsigned long addr)
{
        struct vm_area_struct *vma;
        struct mm_struct *mm = current->mm;

        down_read(&mm->mmap_sem);
        vma = find_vma(mm, addr);
        up_read(&mm->mmap_sem);
        return vma;
}

#define is_page_aligned(x) (!((x) &(PAGE_SIZE - 1)))
static int hog_remap(struct vm_area_struct* vma,
                     unsigned long start, unsigned long size)
{
        unsigned long npages = size>>PAGE_SHIFT;
        struct pages_pool_t* pages_pool = hog_pages_new(npages);
        struct page** pages;
        int err = 0;
        int i;
        printk(KERN_NOTICE "hog_remap: start: %lx size: %lx\n", start, size);
        printk(KERN_NOTICE "npages = 0x%lx\n", npages);
        
        if(!pages_pool)return -ENOMEM;

        pages = pages_pool->pages;
        for(i = 0; i < npages; i++){
                err = vm_insert_page(vma, start + i*PAGE_SIZE, pages[i]);
                if(err){
                        printk(KERN_NOTICE "Failed At: %d\n", i);
                        goto out;
                }
        }
out:        
        hog_pages_del(pages_pool);
        return err;
}

static int hog_ioctl(struct inode * inode, struct file *filp,
                 unsigned int cmd, unsigned long arg)
{
        int err = 0;
        int ok = 1;
        int remain;
        struct hog_area_t area;
        struct vm_area_struct* vma;
        
        if(_IOC_TYPE(cmd) != HOG_IOC_MAGIC) return -EINVAL;
        if(_IOC_NR(cmd) >= HOG_IOC_MAXNR) return -EINVAL;
        
        if(_IOC_DIR(cmd) & _IOC_READ)
                ok = access_ok(VERIFY_WRITE, (void __user *)arg, _IOC_SIZE(cmd));
        else if(_IOC_DIR(cmd) & _IOC_WRITE)
                ok = access_ok(VERIFY_READ, (void __user *)arg, _IOC_SIZE(cmd));
        if(!ok) return -EFAULT;

        switch(cmd) {
        case HOG_IOC_DUMP:
                err = -EPERM;
                current_dump_vma();
                break;
        case HOG_IOC_REMAP:
                remain = copy_from_user(&area, (void*)arg, sizeof(area));
                if(remain != 0)return -EFAULT;
                vma = current_find_vma(area.start);
                if(!vma)return -EFAULT;
                if(!is_page_aligned(area.start) || !is_page_aligned(area.size)
                   || vma->vm_end < area.start+area.size)
                        return -EINVAL;
                err = hog_remap(vma, area.start, area.size);
                break;
        default:
                err = -EINVAL;
                break;
        }
        return err;
}

static int hog_mmap(struct file *filp, struct vm_area_struct *vma)
{
        unsigned long npages = (vma->vm_end - vma->vm_start)>>PAGE_SHIFT;
        struct pages_pool_t* pages_pool = hog_pages_new(npages);
        struct page** pages;
        int err = 0;
        int i;
        printk(KERN_NOTICE "npages = 0x%lx\n", npages);
        
        if(!pages_pool)return -ENOMEM;

        pages = pages_pool->pages;
        for(i = 0; i < npages; i++){
                if (remap_pfn_range(vma, vma->vm_start + i*PAGE_SIZE,
                                    page_to_phys(pages[i])>>PAGE_SHIFT,
                                    PAGE_SIZE,
                                    vma->vm_page_prot)){
                        err = -EAGAIN;
                        goto out;
                }
                
        }
out:        
        hog_pages_del(pages_pool);
        return err;
}

static int __init hog_init(void)
{
        int ret = 0;

        printk("hog_init\n");
        if ((ret = alloc_chrdev_region(&hog_dev, 0, 1, "hog")) < 0) {
                printk(KERN_ERR "could not allocate major number for hog\n");
                goto out;
        }

        cdev_init(&hog_cdev, &hog_fops);
        if ((ret = cdev_add(&hog_cdev, hog_dev, 1)) < 0) {
                printk(KERN_ERR "could not allocate chrdev for hog\n");
                goto out_unalloc_region;
        }
        goto out;
out_unalloc_region:
        unregister_chrdev_region(hog_dev, 1);
out:        
        return ret;
}

static void __exit hog_exit(void)
{
        printk("hog_exit\n");
        cdev_del(&hog_cdev);
        unregister_chrdev_region(hog_dev, 1);
}

module_init(hog_init);
module_exit(hog_exit);
