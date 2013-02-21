#include <stdio.h>

struct A
{
  A(){ printf("A()\n"); }
  ~A() { printf("~A()\n"); }
};

int foo()
{
  static A a;
  return 0;
}
int main()
{
  printf("start.\n");
  foo();
  return 0;
}
