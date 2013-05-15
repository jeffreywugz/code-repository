#include <stdio.h>
#include <locale.h>

int main()
{
  printf("first time: %1$s, second time: %1$s\n", "hello");
  printf("%.*s\n", 2, "hello");
  printf("%.16x\n", 3LL);
  printf("before setlocal: %'d\n", 1<<20);
  setlocale(LC_ALL, "");
  printf("after setlocal: %'d\n", 1<<20);
}
