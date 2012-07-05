#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

int64_t next_pow2_by_clz(int64_t x)
{
  return x <= 2? x: (1LL << 63) >> __builtin_clz(x - 1);
}

int64_t next_pow2(int64_t x)
{
  x -= 1;
  x |= (x >> 1);
  x |= (x >> 2);
  x |= (x >> 4);
  x |= (x >> 8);
  x |= (x >> 16);
  x |= (x >> 32);
  return x + 1;
}

int64_t test_next_pow2(int64_t n)
{
  int64_t s = 0;
  int64_t start = 0;
  int64_t end = 0;
  start = get_usec();
  for(int64_t i = 0; i < n; i++)
    s += next_pow2_by_clz(i);
  end = get_usec();
  printf("next_pow2_by_clz(%ld/%ld=%ld/us)\n", n, end - start, n/(end - start));

  start = get_usec();
  for(int64_t i = 0; i < n; i++)
    s += next_pow2(i);
  end = get_usec();
  printf("next_pow2(%ld/%ld=%ld/us)\n", n, end - start, n/(end - start));
  return s;
}

int main(int argc, char *argv[])
{
  int err = 0;
  if (argc != 2)
  {
    fprintf(stderr, "%s n_items", argv[0]);
  }
  else
  {
    test_next_pow2(atoll(argv[1]));
  }
  return 0;
}
