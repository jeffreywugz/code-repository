#include <phi/kernel.h>

static inline void print_stack(u32 *esp)
{
        int i;
        info("esp: %x\n", esp);
        for(i=0; i<12; i++)
                info("stack top %d: %x\n", i, *esp++);
}

#define ENOSYS 1
int sys_setup(){ return -ENOSYS; }

void sys_exit(int status)
{
}

int sys_read(){ return -ENOSYS; }

int sys_write(char *s)
{
        printf("%s", s);
        return -ENOSYS;
}

int sys_open(const char *file, int mode, int x)
{
        return -ENOSYS;
}

int sys_close(){ return -ENOSYS; }
int sys_waitpid(){ return -ENOSYS; }
int sys_creat(){ return -ENOSYS; }
int sys_link(){ return -ENOSYS; }
int sys_unlink(){ return -ENOSYS; }
int sys_execve(){ return -ENOSYS; }
int sys_chdir(){ return -ENOSYS; }
int sys_time(){ return -ENOSYS; }
int sys_mknod(){ return -ENOSYS; }
int sys_chmod(){ return -ENOSYS; }
int sys_chown(){ return -ENOSYS; }
int sys_break(){ return -ENOSYS; }
int sys_stat(){ return -ENOSYS; }
int sys_lseek(){ return -ENOSYS; }
int sys_getpid(){ return -ENOSYS; }
int sys_mount(){ return -ENOSYS; }
int sys_umount(){ return -ENOSYS; }
int sys_setuid(){ return -ENOSYS; }
int sys_getuid(){ return -ENOSYS; }
int sys_stime(){ return -ENOSYS; }
int sys_ptrace(){ return -ENOSYS; }
int sys_alarm(){ return -ENOSYS; }
int sys_fstat(){ return -ENOSYS; }
int sys_pause(){ return -ENOSYS; }
int sys_utime(){ return -ENOSYS; }
int sys_stty(){ return -ENOSYS; }
int sys_gtty(){ return -ENOSYS; }
int sys_access(){ return -ENOSYS; }
int sys_nice(){ return -ENOSYS; }
int sys_ftime(){ return -ENOSYS; }
int sys_sync(){ return -ENOSYS; }
int sys_kill(){ return -ENOSYS; }
int sys_rename(){ return -ENOSYS; }
int sys_mkdir(){ return -ENOSYS; }
int sys_rmdir(){ return -ENOSYS; }
int sys_dup(){ return -ENOSYS; }
int sys_pipe(){ return -ENOSYS; }
int sys_times(){ return -ENOSYS; }
int sys_prof(){ return -ENOSYS; }
int sys_brk(){ return -ENOSYS; }
int sys_setgid(){ return -ENOSYS; }
int sys_getgid(){ return -ENOSYS; }
int sys_signal(){ return -ENOSYS; }
int sys_geteuid(){ return -ENOSYS; }
int sys_getegid(){ return -ENOSYS; }
int sys_acct(){ return -ENOSYS; }
int sys_phys(){ return -ENOSYS; }
int sys_lock(){ return -ENOSYS; }
int sys_ioctl(){ return -ENOSYS; }
int sys_fcntl(){ return -ENOSYS; }
int sys_mpx(){ return -ENOSYS; }
int sys_setpgid(){ return -ENOSYS; }
int sys_ulimit(){ return -ENOSYS; }
int sys_uname(){ return -ENOSYS; }
int sys_umask(){ return -ENOSYS; }
int sys_chroot(){ return -ENOSYS; }
int sys_ustat(){ return -ENOSYS; }
int sys_dup2(){ return -ENOSYS; }
int sys_getppid(){ return -ENOSYS; }
int sys_getpgrp(){ return -ENOSYS; }
int sys_setsid(){ return -ENOSYS; }
