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

#define is_env_set(key, _default) (0 == (strcmp("true", getenv(key)?: _default)))
#define block_size (1<<9)
#define max_rw_buf_size (4 * block_size)
#define magic 'c'
int64_t g_write_limit_size = 0; // 写线程写到什么位置停止写
int64_t g_n_read_err = 0;
volatile int64_t g_stop = 0;
volatile int64_t g_fsync_pos = 0;
int g_writer_fd = 0;

int write_file_test(const char* path)
{
  int err = 0;
  int fd = 0;
  int64_t pos = 0;
  char* write_buf = NULL;
  int64_t written_len = 0;
  int64_t write_size = 0;
  int open_mode = O_RDWR|O_CREAT|O_TRUNC;
  printf("write_file_test(path=%s)\n", path);
  if (is_env_set("use_append_mode", "true"))
  {
    open_mode |= O_APPEND;
    fprintf(stderr, "write file use append mode\n");
  }
  else
  {
    fprintf(stderr, "write file do not use append mode\n");
  }
  if (is_env_set("write_use_direct_io", "true"))
  {
    open_mode |= O_DIRECT;
    fprintf(stderr, "write use direct io\n");
  }
  else
  {
    fprintf(stderr, "write use buffered io\n");
  }
  
  if (NULL == path)
  {
    err = -EINVAL;
  }
  else if (0 > (fd = open(path, open_mode, S_IRWXU)))
  {
    err = errno;
    perror("open");
  }
  else if (0 != (err = posix_memalign((void**)&write_buf, block_size, max_rw_buf_size)))
  {
    fprintf(stderr, "posix_memalign(size=%ld, align=%ld)=>%d\n", max_rw_buf_size, block_size, err);
  }
  else
  {
    memset(write_buf, magic, max_rw_buf_size);
    g_writer_fd = fd;
    fprintf(stderr, "writer open file[%d], write_buf=%p\n", fd, write_buf);
  }
  while(0 == err && pos < g_write_limit_size)
  {
    write_size = ((random() % max_rw_buf_size)/block_size + 1) * block_size;
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
      pos += written_len;
      fsync(fd);
      g_fsync_pos = pos;
      /* fdatasync(fd); */
    }
  }
  fprintf(stderr, "write to %ld\n", pos);
  g_stop = 1;
  if (NULL != write_buf)
  {
    free(write_buf);
  }
  return err;
}

int read_file_test(const char* path)
{
  int err = 0;
  int fd = 0;
  int64_t pos = 0;
  int64_t read_size = 0;
  char buf_for_correct_check[max_rw_buf_size];
  char* read_buf = NULL;
  int open_mode = O_RDONLY;
  bool wait_fsync = is_env_set("wait_fsync", "false");
  printf("read_file(path=%s)\n", path);
  if (is_env_set("read_use_direct_io", "true"))
  {
    open_mode |= O_DIRECT;
    fprintf(stderr, "read use direct io\n");
  }
  else
  {
    fprintf(stderr, "read use buffered io\n");
  }
  if (NULL == path)
  {
    err = -EINVAL;
  }
  else if (0 > (fd = open(path, open_mode)))
  {
    err = errno;
    perror("open");
  }
  else if (0 != (err = posix_memalign((void**)&read_buf, block_size, max_rw_buf_size)))
  {
    fprintf(stderr, "posix_memalign(size=%ld, align=%ld)=>%d\n", max_rw_buf_size, block_size, err);
  }
  else
  {
    memset(buf_for_correct_check, magic, max_rw_buf_size);
    if (is_env_set("share_fd_with_writer", "false"))
    {
      while(g_writer_fd == 0)
        ;
      fd = g_writer_fd;
    }
    fprintf(stderr, "reader open file[%d], read_buf=%p\n", fd, read_buf);
  }
  while(0 == err && !g_stop)
  {
    if (wait_fsync && pos >= g_fsync_pos)
    {
      fprintf(stderr, "read to end[pos=%ld], need wait\n", g_fsync_pos);
    }
    else if (0 > (read_size = pread(fd, read_buf, max_rw_buf_size, pos)))
    {
      err = errno;
      perror("read");
    }
    else
    {
      if(0 != memcmp(buf_for_correct_check, read_buf, read_size))
      {
        __sync_fetch_and_add(&g_n_read_err, 1);
        fprintf(stderr, "read invalid data at pos=%ld, size=%ld\n", pos, read_size);
      }
      pos += read_size;
    }
  }
  fprintf(stderr, "read to %ld\n", pos);
  if (NULL != read_buf)
  {
    free(read_buf);
  }
  return err;
}

typedef void*(*pthread_handler_t)(void*);
int test_async_rw(const char* path, int64_t n_thread, int64_t n_block)
{
  int err = 0;
  int64_t ret = 0;
  pthread_t thread[n_thread];
  g_write_limit_size = n_block * block_size;
  fprintf(stderr, "test_async_rw(path=%s, n_thread=%ld, n_block=%ld)\n", path, n_thread, n_block);
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
  fprintf(stderr, "total_read_err=%ld\n", g_n_read_err);
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
    err = test_async_rw(getenv("path")?: path, atoll(getenv("n_thread")?:"2")?:2, atoll(argv[1]));
  }
  if (show_help)
  {
    fprintf(stderr, "Usages:\n\tpath=/path/to/file %1$s n_block\n", argv[0]);
  }
  return err;
}
