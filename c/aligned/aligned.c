#include <stdint.h>
#include <stdio.h>

struct A
{
  int64_t x __attribute__((aligned(64)));
  //char c __attribute__((aligned(64)));
};

struct B
{
  char b;
  struct A a;
  char c;
};
  

int main()
{
  printf("sizeof(A)=%ld\n", sizeof(struct A));
  printf("sizeof(B)=%ld\n", sizeof(struct B));
  return 0;
}
