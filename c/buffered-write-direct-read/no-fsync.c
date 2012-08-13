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

#define array_len(A) (sizeof(A)/sizeof(A[0]))
#define is_env_set(key, _default) (0 == (strcmp("true", getenv(key)?: _default)))
#define str_bool(x) (x?"true":"false")
#define block_size (1<<9)
#define max_rw_buf_size ((1<<10) * block_size)
#define magic 'c'

int read_after_write(const char* path, int64_t write_limit_size)
{
  int err = 0;
  int wfd = 0;
  int rfd = 0;
  int64_t n_err = 0;
  int64_t pos = 0;
  char* write_buf = NULL;
  char* read_buf = NULL;
  int64_t write_size = max_rw_buf_size;
  int64_t written_len = 0;
  int64_t read_len = 0;
  bool need_fsync = is_env_set("need_fsync", "false");
  fprintf(stderr, "need_fsync=%s\n", str_bool(need_fsync));
  if (NULL == path)
  {
    err = -EINVAL;
  }
  else if (0 > (wfd = open(path, O_WRONLY|O_CREAT|O_TRUNC, S_IRWXU)))
  {
    err = errno;
    perror("open");
  }
  else if (0 > (rfd = open(path, O_RDONLY|O_DIRECT, S_IRWXU)))
  {
    err = errno;
    perror("open");
  }
  else if (0 != (err = posix_memalign((void**)&write_buf, block_size, max_rw_buf_size)))
  {
    fprintf(stderr, "posix_memalign(size=%ld, align=%ld)=>%d\n", max_rw_buf_size, block_size, err);
  }
  else if (0 != (err = posix_memalign((void**)&read_buf, block_size, max_rw_buf_size)))
  {
    fprintf(stderr, "posix_memalign(size=%ld, align=%ld)=>%d\n", max_rw_buf_size, block_size, err);
  }
  else
  {
    memset(write_buf, magic, max_rw_buf_size);
    memset(read_buf, 0, max_rw_buf_size);
  }
  for(pos = 0; 0 == err && pos < write_limit_size; pos += (written_len > 0)? written_len: 0)
  {
    //fprintf(stderr, "write(pos=%ld)\n", pos);
    if (0 >= (written_len = pwrite(wfd, write_buf, write_size, pos)))
    {
      err = errno;
      perror("write");
      fprintf(stderr, "pwrite(fd=%d, write_size=%ld, pos=%ld) error\n", wfd, write_size, pos);
    }
    else if (written_len != write_size)
    {
      fprintf(stderr, "written_len[%ld] != write_size[%ld]\n", written_len, write_size);
    }
    else if (need_fsync && 0 != fsync(wfd))
    {
      err = errno;
      perror("fsync");
      fprintf(stderr, "fsync(fd=%d, write_size=%ld, pos=%ld) error\n", wfd, write_size, pos);
    }
    else if (0 >= (read_len = pread(rfd, read_buf, write_size, pos)))
    {
      err = errno;
      perror("pread");
      fprintf(stderr, "pread(fd=%d, pos=%ld) error\n", rfd, pos);
    }
    else if (read_len != write_size)
    {
      n_err++;
      fprintf(stderr, "read_len[%ld] != write_size[%ld]\n", read_len, write_size);
    }
    else if (0 != memcmp(read_buf, write_buf, write_size))
    {
      n_err++;
      fprintf(stderr, "memcmp(read_buf != read_buf)\n");
    }
  }
  if (wfd >= 0)
  {
    close(wfd);
  }
  if (rfd >= 0)
  {
    close(rfd);
  }
  if (NULL != write_buf)
  {
    free(write_buf);
  }
  if (NULL != read_buf)
  {
    free(read_buf);
  }
  fprintf(stderr, "read nothing after write happened %ld times\n", n_err);
  return err;
}

int main(int argc, char *argv[])
{
  bool show_help = false;
  int err = 0;
  const char* path = "read-after-write.tmp";
  if (argc != 2)
  {
    err = -EINVAL;
    show_help = true;
  }
  else
  {
    err = read_after_write(getenv("path")?: path, atoll(argv[1]));
  }
  if (show_help)
  {
    fprintf(stderr, "Usages:\n\tpath=/path/to/file %1$s write_size\n", argv[0]);
  }
  return err;
}
