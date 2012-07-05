#define _XOPEN_SOURCE 600
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int test_malloc_align(int64_t n, int64_t size, int64_t align)
{
  int err = 0;
  char* p = NULL;
  if (0 >= n || 0 >= size || 0 >= align)
  {
    err = -EINVAL;
  }
  for(int64_t i = 0; 0 == err && i < n; i++)
  {
    if (0 != (err = posix_memalign((void**)&p, align, size)))
    {
      fprintf(stderr, "posix_memalign(align=%ld, size=%ld)=>%d", align, size, err);
    }
    /* if (NULL == (p = malloc(size))) */
    /* { */
    /*   fprintf(stderr, "malloc(size=%ld)=>NULL\n", size); */
    /* } */
    else
    {
      free(p);
    }
  }
  return err;
}

int main(int argc, char *argv[])
{
  int err = 0;
  int64_t block_size = 1<<20;
  int64_t align = block_size;
  if (argc != 2)
  {
    fprintf(stderr, "%s n_block\n", argv[0]);
  }
  else
  {
    err = test_malloc_align(atoll(argv[1]), block_size, align);
  }
  return err;
}
