#define _GNU_SOURCE
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

int server_start(const char* addr)
{
}

int client_start(const char* addr, const int64_t n_client, const int64_t packet_size, const int64_t duration)
{
}

int get_disk_param(const char* path, int64_t size)
{
  int err = 0;
  int fd = 0;
  int64_t total_read_ops = size/(block_size * n_blocks);
  int64_t blk_idx = 0;
  int64_t start_time = 0;
  int64_t end_time = 0;
  printf("get_disk_bw(path=%s, size=%ld)\n", path, size);
  if (NULL == path || 0 >= size)
  {
    err = -EINVAL;
  }
  else if (0 > (fd = open(path, O_RDWR|O_CREAT|O_DIRECT|O_TRUNC, S_IRWXU)))
  {
    err = errno;
    perror("open");
  }
  start_time = get_usec();
  for(int64_t s = 0; 0 == err && s < size; s += sizeof(file_block))
  {
    if (0 >= write(fd, file_block, sizeof(file_block)))
    {
      err = errno;
      perror("write");
    }
    else
    {
      /* sync(); */
      /* fdatasync(fd); */
    }
  }
  end_time = get_usec();
  printf("write bw: %ldB/%ldus = %.2fMB/s\n", size, end_time - start_time, 1.0 * size/(end_time - start_time + 1));

  lseek(fd, 0, SEEK_SET);
  start_time = get_usec();
  for(int64_t s = 0; 0 == err && s < size; s += sizeof(file_block))
  {
    if (0 >= read(fd, file_block, sizeof(file_block)))
    {
      err = errno;
      perror("read");
    }
  }
  end_time = get_usec();
  printf("read bw: %ldB/%ldus = %.2fMB/s\n", size, end_time - start_time, 1.0 * size/(end_time - start_time + 1));

  lseek(fd, 0, SEEK_SET);
  start_time = get_usec();
  for(int64_t s = 0; 0 == err && s < total_read_ops; s++)
  {
    blk_idx = random() % (size/block_size);
    lseek(fd, blk_idx * block_size, SEEK_SET);
    if (0 >= read(fd, file_block, block_size))
    {
      err = errno;
      perror("read");
    }
  }
  end_time = get_usec();
  printf("read ops: %ld/%ldus = %.2fOPS/s\n", total_read_ops, end_time - start_time, 1e6 * total_read_ops/(end_time - start_time + 1));

  if (fd >= 0)
    close(fd);
  return err;
}

int64_t get_file_size(int fd)
{
  int err = 0;
  int64_t size = 0;
  struct stat st;
  if (0 != (err = fstat(fd, &st)))
  {
    perror("fstat");
  }
  else
  {
    size = st.st_size;
  }
  return size;
}

int get_iops(const char* path, int64_t total_read_ops)
{
  int err = 0;
  int fd = 0;
  int64_t size = 0;
  int64_t blk_idx = 0;
  int64_t start_time = 0;
  int64_t end_time = 0;
  printf("get_disk_iops(path=%s, total_read_ops=%ld)\n", path, total_read_ops);
  if (NULL == path || 0 >= total_read_ops)
  {
    err = -EINVAL;
  }
  else if (0 > (fd = open(path, O_RDONLY|O_DIRECT)))
  {
    err = errno;
    perror("open");
  }
  else if (0 != ioctl(fd, BLKGETSIZE64, &size))
  {
    err = errno;
    perror("ioctl");
  }
  else
  {
    printf("%s size=%ld\n", path, size);
  }
  start_time = get_usec();
  for(int64_t s = 0; 0 == err && s < total_read_ops; s++)
  {
    blk_idx = random() % (size/block_size);
    lseek(fd, blk_idx * block_size, SEEK_SET);
    if (0 >= read(fd, file_block, block_size))
    {
      err = errno;
      perror("read");
    }
  }
  end_time = get_usec();
  printf("read ops: %ld/%ldus = %.2fOPS/s\n", total_read_ops, end_time - start_time, 1e6 * total_read_ops/(end_time - start_time + 1));

  lseek(fd, 0, SEEK_SET);
  return err;
}

int main(int argc, char *argv[])
{
  bool show_help = false;
  int err = 0;
  const char* path = "disk-param.tmp";
  const char* disk = "/dev/sda";
  if (argc != 3)
  {
    err = -EINVAL;
    show_help = true;
  }
  else if (0 == strcmp(argv[1], "bw"))
  {
    err = get_disk_param(getenv("path")?: path, atoll(argv[2]));
  }
  else if (0 == strcmp(argv[1], "iops"))
  {
    err = get_iops(getenv("path")?: disk, atoll(argv[2]));
  }
  else
  {
    err = -EINVAL;
    show_help = true;
  }
  if (show_help)
  {
    fprintf(stderr, "Usages:\n" "\tpath=/path/to/file %1$s bw size\n" "\tpath=/dev/sda1 %1$s iops n_read_ops\n", argv[0]);
  }
  return err;
}
