#ifndef MBOOT_H
#define MBOOT_H

struct mboot_info
{
        u32 flag;
        u32 mem_lower;
        u32 mem_upper;
        u32 boot_device;
        u32 cmdline;
        u32 mods_count;
        u32 mods_addr;
        u32 aout_sym_elf_sec[4];
        u32 mmap_length;
        u32 mmap_addr;
} __attribute__((packed));

struct mboot_mmap
{
	u32 size;
        u64 base;
        u64 len;
	u32 type;
} __attribute__((packed));

#endif //MBOOT_H
