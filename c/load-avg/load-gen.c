#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
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

void* cpu_loop(void* arg)
{
  while(1)
    ;
  return NULL;
}

void* sleep_loop(void* arg)
{
  while(1)
    usleep(1000000);
  return NULL;
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

void* io_loop(void* arg)
{
  const int64_t n_read_ops = 1<<12;
  while(1)
    get_iops(getenv("disk_path")?: "/dev/sda", n_read_ops);
  return NULL;
}

typedef void*(*pthread_handler_t)(void*);
int gen_load(const char* type, const int64_t n_threads)
{
  pthread_t threads[n_threads];
  pthread_handler_t handler = NULL;
  if (0 == strcmp(type, "cpu")){
    handler = cpu_loop;
  } else if (0 == strcmp(type, "sleep")) {
    handler = sleep_loop;
  } else if (0 == strcmp(type, "disk")) {
    handler = io_loop;
  }
  for (int64_t i = 0; i < n_threads; i++){
    pthread_create(threads + i, NULL, (pthread_handler_t)handler, NULL);
  }
  for (int64_t i = 0; i < n_threads; i++){
    pthread_join(threads[i], NULL);
  }
  return 0;
}

int main(int argc, char *argv[])
{
  int err = 0;
  if (argc != 3) {
    err = -EINVAL;
    fprintf(stderr, "Usages:\n" "\t./load-gen [sleep|cpu|disk] n_threads\n");
  } else {
    err = gen_load(argv[1], atoll(argv[2]));
  }
  return err;
}
