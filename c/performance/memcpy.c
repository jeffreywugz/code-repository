#include <errno.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

#define profile(expr) ({                        \
  int64_t old_us = 0;                                  \
  int64_t new_us = 0;                                  \
  int64_t result = 0;                                    \
  old_us = get_usec();                                   \
  result = expr;                                         \
  new_us = get_usec();                                                  \
  printf("%s=>%ld in %lfms\n", #expr, result, 1.0 * (new_us - old_us)/1000);       \
  new_us - old_us; })

// no check
void* memcpy_byte_by_byte(char* dest, const char* src, int64_t size)
{
  for(int64_t i = 0; i < size; i++)
    *dest++ = *src++;
  return dest;
}

// no check
void* memcpy_int_by_int(char* dest, const char* src, int64_t size)
{
  int64_t* dest2 = (int64_t*)dest;
  const int64_t* src2 = (const int64_t*)src;
  for(int64_t i = 0; i < size/sizeof(int64_t); i++)
    *dest2++ = *src2++;
  return dest;
}

int test_memcpy(const int64_t size)
{
  int err = 0;
  char* src = malloc(size);
  char* dest = malloc(size);
  assert(src && dest);
  memset(src, 0, size);
  memset(dest, 0, size);
  profile(memcpy(src, dest, size));
  profile(memcpy_byte_by_byte(src, dest, size));
  profile(memcpy_int_by_int(src, dest, size));
  if(src)free(src);
  if(dest)free(dest);
  return err;
}

int main(int argc, char *argv[])
{
  int err = 0;
  bool show_help = false;
  if (argc != 2)
  {
    show_help = true;
  }
  else if (0 != (err = test_memcpy(atoll(argv[1]))))
  {
    fprintf(stderr, "test_memcpy(%s)=>%d\n", argv[1], err);
  }
  if (show_help)
  {
    fprintf(stderr, "Usages:\n\t%s size", argv[1]);
  }
  return err;
}
