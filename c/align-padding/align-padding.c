#include <stdint.h>
#include <stdio.h>

int64_t get_align_padding(const int64_t x, const int64_t mask)
{
  return -x & mask;
}

#define test_func(expr) printf("%s=>%ld\n", #expr, expr)
int main(int argc, char *argv[])
{
  test_func(get_align_padding(3, 8-1));
  test_func(get_align_padding(8, 8-1));
  test_func(get_align_padding(11, 8-1));
  test_func(get_align_padding(24, 8-1));
  test_func(get_align_padding(-23, 8-1));
  return 0;
}
