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
#include <locale.h>

#define cfg(key, default_value) (getenv(key)?:default_value)
#define cfgi(key, default_value) atoll(cfg(key, default_value))

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

struct exp_stat_t
{
  int64_t count[64];
  int64_t time[64];
};

int exp_stat_init(struct exp_stat_t* stat)
{
  memset(stat, 0, sizeof(*stat));
  return 0;
}

int64_t exp_get_idx(const int64_t x)
{
  return x? (64 - __builtin_clzl(x)): 0;
}

int exp_stat_add(struct exp_stat_t* stat, int64_t x)
{
  int64_t idx = exp_get_idx(x);
  stat->count[idx]++;
  stat->time[idx]+=x;
}

int exp_stat_report(struct exp_stat_t* stat)
{
  int64_t total_count = 0;
  int64_t total_time = 0;
  for(int64_t i = 0; i < 64; i++)
  {
    total_count += stat->count[i];
    total_time += stat->time[i];
  }
  printf("total_count=%'ld, total_time=%'ldus, avg=%'ldus\n", total_count, total_time, total_time/total_count);
  for(int64_t i = 0; i < 32; i++)
  {
    printf("stat[..<latency<2**%d]: %lf%%, count=%'ld, time=%'ldus, avg=%'ldus\n",
           i, 100.0*stat->count[i]/(total_count+1), stat->count[i], stat->time[i], stat->count[i] == 0? 0: stat->time[i]/stat->count[i]);
  }
  return 0;
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
  struct exp_stat_t stat;
  int64_t file_size = cfgi("file_size","64");
  int64_t FILE_COUNT = cfgi("total_size", "8000")/file_size;
  int64_t time_limit = cfgi("time_limit", "3000000");
  int64_t write_count = 0;
  int64_t end_time = 0;
  int64_t cur_start_time = 0;
  int64_t cur_end_time = 0;
  int64_t file_no = 0;
  char file_path[256];
  end_time = get_usec() + time_limit;
  printf("test_disk_latency(path=%s, sleep_us=%ld, write_size=%ld, time_limit=%ld, end_time=%ld, file_size=%ld)\n",
         path, sleep_us, write_size, time_limit, end_time, file_size);
  exp_stat_init(&stat);
  if (NULL == path || 0 >= write_size || write_size > sizeof(file_block) || 0 >= sleep_us)
  {
    err = -EINVAL;
  }
  cur_start_time = cur_end_time = get_usec();
  for(write_count = 0; 0 == err && cur_start_time < end_time; write_count++)
  {
    cur_start_time = cur_end_time;
    if (fd > 0)
    {}
    // 最开始创建20个文件，然后重用文件
    else if (0 >= snprintf(file_path, sizeof(file_path), "%s/%ld", path, file_no%FILE_COUNT))
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
      fsync(fd);
    }
    if (fd > 0 && write_count * write_size > (file_no + 1) * file_size* 1024*1024)
    {
      close(fd);
      fd = 0;
      file_no++;
      if (0 == (file_no % 100))
      {
        printf("switch_file(%ld): pass %ld\n", file_no, (write_count * write_size) / (FILE_COUNT * file_size*1024*1024));
      }
    }
    cur_end_time = get_usec();
    //printf("write: time=<%ld:%ld>:%ld\n", cur_start_time, cur_end_time, cur_end_time-cur_start_time);
    exp_stat_add(&stat, cur_end_time - cur_start_time);
    if (sleep_us > 0)
    {
      usleep(sleep_us);
    }
  }
  exp_stat_report(&stat);
  return err;
}

int main(int argc, char *argv[])
{
  bool show_help = false;
  int err = 0;
  const char* path = ".";
  setlocale(LC_ALL, "");
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
    fprintf(stderr, "Usages:\n" "\tpath=/path/to/dir total_size=80000 file_size=60 time_limit=30000000 %1$s sleep_us write_size\n", argv[0]);
  }
  return err;
}
