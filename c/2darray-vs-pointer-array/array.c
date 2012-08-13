#include <stdio.h>

int* a11_addr1(int (*a)[16])
{
  return &a[1][1];
}

int* a11_addr2(int *a[16])
{
  return &a[1][1];
}

int* a11_addr3(int* p, int size)
{
  int (*a)[size] = p;
  return &a[1][1];
}

int main()
{
  int err = 0;
  int* p = 0;
  return err;
}
