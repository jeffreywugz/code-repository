#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

int timing_memset(int64_t total_size, int64_t n_block)
{
  char* buf = malloc(total_size);
  int64_t before_us = get_usec();
  int64_t after_us = 0;
  for(int64_t offset = 0; offset < total_size; offset += total_size/n_block)
    memset(buf + offset, 0xff, total_size/n_block);
  after_us = get_usec();
  free(buf);
  return after_us - before_us;
}

int main(int argc, char** argv)
{
  int err = 0;
  if (argc != 3)
  {
    err = -EINVAL;
    fprintf(stderr, "Usages: ./memset total_size n_block\n");
  }
  else
  {
    printf("memset(total_size=%s, n_block=%s): %lf ms\n",
           argv[1], argv[2], 1.0 * timing_memset(atoll(argv[1]), atoll(argv[2]))/1000);
  }
  return err;
}
