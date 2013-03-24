#include <stdio.h>

int main()
{
  char buf1[512] __attribute__((aligned(512)));
  static char buf2[512] __attribute__((aligned(512)));
  printf("stack array addr: %p, static array addr: %p\n", buf1, buf2);
  return 0;
}
