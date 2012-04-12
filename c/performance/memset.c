#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

void* my_memset(void* s, int c, size_t n)
{
  for(size_t i = 0; i < n; i++)
    *((char*)s + i) = c;
  return s;
}

typedef void* (*memset_func_t)(void* s, int c, size_t n);
int test_memset(int64_t total_size, int64_t n_block)
{
  char* buf = NULL;
  int64_t after_us = 0;
  int64_t before_us = 0;
  char* prefault = getenv("prefault");
  char* use_my_memset = getenv("use_my_memset");
  memset_func_t memset_func = use_my_memset && (0 == strcmp("true", use_my_memset))? my_memset: memset;
  
  printf("test_memset(total_size=%ld, n_block=%ld, prefault=%s, use_my_memset=%s):\n",
         total_size, n_block, prefault, use_my_memset);

  if (total_size <= 0 || n_block <= 0 || n_block > total_size || total_size%n_block)
    return -EINVAL;
  if (NULL == (buf = malloc(total_size)))
    return -ENOMEM;

  before_us = get_usec();
  for(int64_t offset = 0; prefault && (0 == strcmp(prefault, "true")) && offset < total_size; offset += getpagesize())
    *(buf + offset) = 0xff;
  after_us = get_usec();
  printf("pre fault: %lf ms\n", (after_us-before_us)/1000.0);

  for(int64_t k = 0; k < 2; k++) {
    before_us = get_usec();
    for(int64_t offset = 0; offset < total_size; offset += total_size/n_block)
      memset_func(buf + offset, 0xff, total_size/n_block);
    after_us = get_usec();
    printf("iter %ld, time = %lf ms\n", k, (after_us-before_us)/1000.0);
  }

  free(buf);
  return 0;
}

int main(int argc, char** argv)
{
  int err = 0;
  if (argc != 3)
  {
    err = -EINVAL;
    fprintf(stderr, "Usages: prefault=true use_my_memset=true ./memset total_size n_block\n");
  }
  else if (0 != (err =test_memset(atoll(argv[1]), atoll(argv[2]))))
  {
    fprintf(stderr, "test_memset()=>%d\n", err);
  }
  return err;
}
