#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    printf("./malloc size\n");
    return -1;
  }
  else if(NULL != malloc(atoll(argv[1])))
  {
    printf("alloc %ld succ\n"), atoll(argv[1]);
  }
  else
  {
    printf("alloc %ld fail\n"), atoll(argv[1]);
  }
  malloc_stats();
  return 0;
}
