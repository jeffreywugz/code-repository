// cflags = -lpthread -lrt -std=c99 -g -O2 -D_GNU_SOURCE
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

int64_t test_disk_latency(const char* path, int64_t sleep_us, int64_t write_size)
{
  int err = 0;
  int fd = 0;
  int64_t write_count = 0;
  int64_t end_fsync_time = 0;
  int64_t end_fdatasync_time = 0;
  int64_t end_write_time = 0;
  int64_t time_limit = 0;
  int64_t total_time = 0;
  int64_t start_time = 0;
  int64_t end_time = 0;
  int64_t file_no = 0;
  int64_t file_size = 0;
  char file_path[256];
  time_limit = atoll(getenv("time_limit")?:"3000000");
  file_size = atoll(getenv("file_size")?:"60");
  printf("test_disk_latency(path=%s, sleep_us=%ld, write_size=%ld, time_limit=%ld, file_size=%ld)\n",
         path, sleep_us, write_size, time_limit, file_size);
  if (NULL == path || 0 >= write_size || write_size > sizeof(file_block) || 0 >= sleep_us)
  {
    err = -EINVAL;
  }
  // 在前1/3时间调用fsync，中间1/3时间调用fdatasync，最后1/3时间什么都不调用
  end_fsync_time = get_usec() + time_limit/3;
  end_fdatasync_time = get_usec() + 2*time_limit/3;
  end_write_time = get_usec() + time_limit;
  for(write_count = 0; 0 == err ; write_count++)
  {
    start_time = get_usec();
    if (start_time > end_write_time)break;
    if (fd > 0)
    {}
    // 最开始创建20个文件，然后重用文件
    else if (0 >= snprintf(file_path, sizeof(file_path), "%s/%ld", path, file_no%20))
    {}
    else if (0 > (fd = open(file_path, O_RDWR|O_CREAT|O_DIRECT, S_IRWXU)))
    {
      err = errno;
      perror("open");
    }
    if (0 >= pwrite(fd, file_block, write_size, (write_count * write_size)  % (file_size*1024*1024)))
    {
      err = errno;
      perror("write");
    }
    else
    {
      if (start_time < end_fsync_time || true)
      {
        fsync(fd);
      }
      else if (start_time < end_fdatasync_time)
      {
        fdatasync(fd);
      }
      else
      {
      }
    }
    if (fd > 0 && write_count * write_size > (file_no + 1) * file_size* 1024*1024)
    {
      close(fd);
      fd = 0;
      file_no++;
      printf("switch_file(%ld)\n", file_no);
    }
    end_time = get_usec();
    printf("write: time=<%ld:%ld>:%ld\n", start_time, end_time, end_time-start_time);
    total_time += end_time - start_time;
    usleep(sleep_us);
  }
  printf("write latency: %ldus/%ld = %ldus\n", total_time, write_count, write_count? total_time/write_count: 0);
  return err;
}

int main(int argc, char *argv[])
{
  bool show_help = false;
  int err = 0;
  const char* path = ".";
  if (argc != 3)
  {
    err = -EINVAL;
    show_help = true;
  }
  else
  {
    err = test_disk_latency(getenv("path")?: path, atoll(argv[1]), atoll(argv[2]));
  }
  if (show_help)
  {
    fprintf(stderr, "Usages:\n" "\tpath=/path/to/dir time_limit=30000000 %1$s sleep_us write_size\n", argv[0]);
  }
  return err;
}
