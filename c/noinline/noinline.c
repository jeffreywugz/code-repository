#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int64_t __attribute__((noinline)) mul(int64_t x, int64_t y)
{
  return x * y;
}

int64_t  test_noinline(int64_t n)
{
  int64_t x = 0;
  for(int64_t i = 0; i < n; i++)
  {
    x += mul(3, 4);
  }
  return x;
}

int main(int argc, char *argv[])
{
  int err = 0;
  if (argc != 2)
  {
    err = -EINVAL;
    fprintf(stderr, "%s n\n", argv[0]);
  }
  else
  {
    //test_noinline(atoll(argv[1]));
    printf("test_noinline: %ld\n", test_noinline(atoll(argv[1])));
  }
  return err;
}
