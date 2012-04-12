#include <sys/time.h>
int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

#define profile(expr) ({                                                \
      int64_t old_us = 0;                                               \
      int64_t new_us = 0;                                               \
      int64_t result = 0;                                               \
      old_us = get_usec();                                              \
      result = expr;                                                    \
      new_us = get_usec();                                              \
      printf("%s=>%ld in %ldms\n", #expr, result, new_us - old_us);     \
      new_us - old_us; })

int main(int argc, char *argv[])
{
  int64_t n = argc > 1 ? atoi(argv[1]): 0;
  n = (0 == n)? 10: n;
  return 0;
}
