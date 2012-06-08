#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    fprintf(stderr, "./alloca alloca size\n" "./alloca malloc size\n");
    return -EINVAL;
  }
  else if (0 == strcmp(argv[1], "alloca"))
  {
    assert(alloca(atoll(argv[2])));
    pause();
  }
  else if (0 == strcmp(argv[1], "malloc"))
  {
    assert(malloc(atoll(argv[2])));
    pause();
  }
  return 0;
}
