#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
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

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

#define block_size (1<<9)
#define n_blocks (1<<12)
char file_block[block_size * n_blocks] __attribute__ ((aligned (block_size))) ;

int seq_io(const char* path, int64_t size)
{
  int err = 0;
  int fd = 0;
  int64_t start_time = 0;
  int64_t end_time = 0;
  printf("seq_io(path=%s, size=%ld)\n", path, size);
  if (NULL == path || 0 >= size)
  {
    err = -EINVAL;
  }
  else if (0 > (fd = open(path, O_RDWR|O_CREAT|O_DIRECT, S_IRWXU)))
  {
    err = errno;
    perror("open");
  }
  else if (0 != ftruncate(fd, size))
  {
    err = errno;
    perror("ftrucate");
  }
  //else if (0 != posix_fadvise(fd, 0, size, POSIX_FADV_SEQUENTIAL|POSIX_FADV_NOREUSE))
  //else if (0 != posix_fadvise(fd, 0, size, POSIX_FADV_NOREUSE))
  else if (false)
  {
    err = errno;
    perror("fadvise");
  }
  start_time = get_usec();
  for(int64_t s = 0; 0 == err && s < size; s += block_size)
  {
    if (0 >= write(fd, file_block, block_size))
    {
      err = errno;
      perror("read");
    }
  }
  end_time = get_usec();
  printf("write bandwidth: %ld/%ldus = %.2fM/s\n", size, end_time - start_time, 1.0 * size/(end_time - start_time + 1));

  lseek(fd, 0, SEEK_SET);
  start_time = get_usec();
  for(int64_t s = 0; 0 == err && s < size; s += block_size)
  {
    if (0 >= read(fd, file_block, block_size))
    {
      err = errno;
      perror("read");
    }
  }
  end_time = get_usec();
  printf("read bandwidth: %ld/%ldus = %.2fM/s\n", size, end_time - start_time, 1.0 * size/(end_time - start_time + 1));

  return err;
}

int main(int argc, char *argv[])
{
  bool show_help = false;
  int err = 0;
  const char* path = "fadvise-seq.tmp";
  if (argc != 2)
  {
    err = -EINVAL;
    show_help = true;
  }
  else if (0 != (err = seq_io(getenv("path")?: path, atoll(argv[1]))))
  {
    fprintf(stderr, "seq_io(%s)=>%d\n", getenv("path")?:path, err);
  }
  if (show_help)
  {
    fprintf(stderr, "Usages:\n" "\tpath=/path/to/file %1$s size\n", argv[0]);
  }
  return err;
}
