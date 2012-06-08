#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
  for(int i = 0; i < argc; i++)
  {
    printf("argv[%d]=%s,len=%ld\n", i, argv[i], strlen(argv[i]));
  }
  return 0;
}
