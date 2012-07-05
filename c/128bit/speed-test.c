#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

int speed_test(int64_t n)
{
  int64_t start = 0;
  int64_t end = 0;
  __uint64_t i64 = 0;
  __uint64_t s64 = 0;
  __uint64_t i128 = 0;
  __uint128_t s128 = 0;
  start = get_usec();
  for(int64_t i = 0; i < n; i++)
    s64 ^= ++i64;
  end = get_usec();
  printf("calc with __uint64_t[loop=%ld] use %dms\n", n, (end - start)/1000);
  start = get_usec();
  for(int64_t i = 0; i < n; i++)
    s128 ^= ++i128;
  end = get_usec();
  printf("calc with __uint128_t[loop=%ld] use %dms\n", n, (end - start)/1000);
  return 0;
}

int main(int argc, char *argv[])
{
  int err = 0;
  int64_t n =  0;
  if (argc != 2)
  {
    err = -EINVAL;
    fprintf(stderr, "%s n", argv[0]);
  }
  else
  {
    err = speed_test(1<<atoll(argv[1]));
  }
  return 0;
}
