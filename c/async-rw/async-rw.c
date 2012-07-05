//#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <pthread.h>

#define array_len(A) (sizeof(A)/sizeof(A[0]))
int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

#define block_size (1<<9)
#define magic 'c'
int64_t g_write_limit_size = 0;
int64_t g_err = 0;
volatile int64_t g_stop = 0;
volatile int64_t g_pos = 0;
char write_buf[16 * block_size] __attribute__ ((aligned (block_size)));

int g_fd = 0;
int write_file_test(const char* path)
{
  int err = 0;
  int fd = 0;
  int64_t pos = 0;
  int64_t written_len = 0;
  int64_t write_size = 0;
  int open_mode = O_RDWR|O_CREAT|O_TRUNC;
  if (getenv("use_append_mode"))
  {
    open_mode |= O_APPEND;
    fprintf(stderr, "write file use append mode\n");
  }
  else
  {
    fprintf(stderr, "write file do not use append mode\n");
  }
  if (!getenv("write_use_buffered_io"))
  {
    open_mode |= O_DIRECT;
    fprintf(stderr, "write use direct io\n");
  }
  else
  {
    fprintf(stderr, "write use buffered io\n");
  }
  
  memset(write_buf, magic, sizeof(write_buf));
  printf("write_file_test(path=%s)\n", path);
  if (NULL == path)
  {
    err = -EINVAL;
  }
  else if (0 > (fd = open(path, open_mode, S_IRWXU)))
  {
    err = errno;
    perror("open");
  }
  else
  {
    g_fd = fd;
    fprintf(stderr, "open file[%d]\n", g_fd);
  }
  while(0 == err && pos < g_write_limit_size)
  {
    write_size = ((random() % sizeof(write_buf))/block_size + 1) * block_size;
    if (0 >= (written_len = pwrite(fd, write_buf, write_size, pos)))
    {
      err = errno;
      perror("write");
      fprintf(stderr, "pwrite(fd=%d, write_size=%ld, pos=%ld) error\n", fd, write_size, pos);
    }
    else if (written_len != write_size)
    {
      fprintf(stderr, "written_len[%ld] != write_size[%ld]\n", written_len, write_size);
    }
    else
    {
      pos += write_size;
      fsync(fd);
      g_pos = pos;
      /* fdatasync(fd); */
    }
  }
  fprintf(stderr, "write to %ld\n", pos);
  g_stop = 1;
  return err;
}

char read_buf[16 * block_size] __attribute__ ((aligned (block_size)));
int read_file_test(const char* path)
{
  int err = 0;
  static int fd = 0;
  int local_fd = 0;
  int64_t pos = 0;
  int64_t read_size = 0;
  char stdbuf[16 * block_size];
  int open_mode = O_RDONLY;
  if (!getenv("read_use_buffered_io"))
  {
    open_mode |= O_DIRECT;
    fprintf(stderr, "read use direct io\n");
  }
  else
  {
    fprintf(stderr, "read use buffered io\n");
  }
  
  memset(stdbuf, magic, sizeof(stdbuf));
  printf("read_file(path=%s)\n", path);
  if (NULL == path)
  {
    err = -EINVAL;
  }
  else if (0 > (local_fd = open(path, open_mode)))
  {
    err = errno;
    perror("open");
  }
  else if(__sync_bool_compare_and_swap(&fd, 0, local_fd))
  {
    fprintf(stderr, "set fd to local_fd[%d]\n", local_fd);
  }
  while(g_fd == 0)
    ;
  fd = g_fd;
  while(0 == err && !g_stop)
  {
    if (false && pos >= g_pos)
    {
      fprintf(stderr, "read to end[pos=%ld], need wait\n", g_pos);
    }
    else if (0 > (read_size = pread(fd, read_buf, sizeof(read_buf), pos)))
    {
      err = errno;
      perror("read");
    }
    else
    {
      if(0 != memcmp(stdbuf, read_buf, read_size))
      {
        __sync_fetch_and_add(&g_err, 1);
        fprintf(stderr, "read invalid data at pos=%ld\n", pos);
      }
      pos += read_size;
    }
  }
  fprintf(stderr, "read to %ld\n", pos);
  return err;
}

typedef void*(*pthread_handler_t)(void*);
int test_async_rw(const char* path, int64_t n_block)
{
  int err = 0;
  int64_t ret = 0;
  pthread_t thread[3];
  g_write_limit_size = n_block * block_size;
  for (int64_t i = 0; i < array_len(thread); i++) {
    if (i == 0)
    {
      pthread_create(thread + i, NULL, (pthread_handler_t)write_file_test, (void*)path);
    }
    else
    {
      pthread_create(thread + i, NULL, (pthread_handler_t)read_file_test, (void*)path);
    }
  }
  for (int64_t i = 0; i < array_len(thread); i++) {
    ret = 0;
    pthread_join(thread[i], (void**)&ret);
    if (ret != 0)
    {
      fprintf(stderr, "thread[%ld] err=%ld\n", i, ret);
    }
  }
  fprintf(stderr, "total_read_err=%ld\n", g_err);
  return err;
}

int main(int argc, char *argv[])
{
  bool show_help = false;
  int err = 0;
  const char* path = "async-rw.tmp";
  if (argc != 2)
  {
    err = -EINVAL;
    show_help = true;
  }
  else
  {
    err = test_async_rw(getenv("path")?: path, atoll(argv[1]));
  }
  if (show_help)
  {
    fprintf(stderr, "Usages:\n\tpath=/path/to/file %1$s n_block\n", argv[0]);
  }
  return err;
}
