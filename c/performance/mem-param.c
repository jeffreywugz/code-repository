#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>
#include <assert.h>

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

#define unit 8
int mem_sweep(int64_t size, int64_t stride, int64_t n_access)
{
  int64_t start_time = 0;
  int64_t end_time = 0;
  int64_t s = 0;
  int64_t *p = 0;
  char* buf = malloc(size);
  assert(buf);
  memset(buf, 0, size);
  start_time = get_usec();
  for(int64_t j = 0; j < (n_access*stride/size); j++) {
    for(int64_t i = 0; i < size; i += stride) {
      s |= buf[i + j % stride];
    }
  }
  end_time = get_usec();
  printf("s = %ld\n", s);
  printf("size=%ld, strid=%ld, access_time=%ldus/%ld=%.2fns\n", size, stride, end_time - start_time, n_access, 1e3 * (end_time - start_time)/n_access);
  return 0;
}

int main(int argc, char *argv[])
{
  bool show_help = false;
  int err = 0;
  if (argc != 4)
  {
    err = -EINVAL;
    show_help = true;
  }
  else if (0 != (err = mem_sweep(atoll(argv[1]), atoll(argv[2]), atoll(argv[3]))))
  {
    fprintf(stderr, "mem_sweep()=>%d\n", err);
  }
  if (show_help)
  {
    fprintf(stderr, "Usages:\n" "\t%1$s size stride n_access\n", argv[0]);
  }
  return err;
}
