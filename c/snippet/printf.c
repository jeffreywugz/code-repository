#include <stdio.h>

int main()
{
  printf("first time: %1$s, second time: %1$s\n", "hello");
  printf("%.*s\n", 2, "hello");
  printf("%.16x\n", 3LL);
}
