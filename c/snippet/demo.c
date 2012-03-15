#include <stdio.h>
#include <unistd.h>

int main(int argc, char** argv)
{
  int i = 1;
  printf("%ld\n", i++>>1);
  return 0;
}
