#include <stdint.h>

uint64_t unsigned_divide(uint64_t x)
{
  return x/1024;
}

int64_t signed_divide(int64_t x)
{
  return x/1024;
}

int main()
{
  uint64_t ux = 1;
  int64_t x = -1;
  printf("%lu/1024=%lu\n%ld/1024=%ld", ux, unsigned_divide(ux), x, signed_divide(x));
  return 0;
}
