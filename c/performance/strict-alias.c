#include <stdint.h>
#include <stdio.h>

struct A
{
  int64_t x;
};
struct B
{
  int64_t x;
};
int set0(struct A* a)
{
  a->x = 0;
}
int main()
{
  struct B b;
  b.x = 1;
  set0(&b);
  printf("%ld\n", b.x);
  return 0;
}
