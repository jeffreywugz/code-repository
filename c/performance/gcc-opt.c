#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}
#define profile(expr) {int64_t start = get_usec(); expr; printf("%s: %ldus\n", #expr, get_usec() - start); }

#define N (1LL<<30)
int for_loop1()
{
  int64_t s = 0;
  for(int64_t i = 0; i < N; i++)
  {
    s |= i;
  }
  return 0;
}

int for_loop2()
{
  int64_t s = 0;
  for(int64_t i = 0; i < N; i++)
  {
    s += i;
  }
  printf("%ld\n", s);
  return 0;
}

int for_loop3()
{
  int64_t s = 0;
  for(int64_t i = 0; i < N; i++)
  {
    s |= i;
  }
  printf("%ld\n", s);
  return 0;
}

int for_loop4()
{
  int64_t s = 0;
  for(int64_t i = 0; i < (1LL<<33); i++)
  {
    s += i;
  }
  printf("%ld\n", s);
  return 0;
}

int main()
{
  profile(for_loop1());
  profile(for_loop2());
  profile(for_loop3());
  profile(for_loop4());
  return 0;
}
