//#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/time.h>

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

struct prof_entry_t
{
  const char* file_;
  const char* func_;
  const int64_t line_;
  int64_t start_us_;
  int64_t end_us_;
};

void print_profile_entry(struct prof_entry_t* pe)
{
  fprintf(stderr, "%s:%ld:%s using %ldus\n", pe->file_, pe->line_, pe->func_, pe->end_us_ - pe->start_us_);
}

static void prof_func_end(struct prof_entry_t *pe)
{
  pe->end_us_ = get_usec();
  print_profile_entry(pe);
}

#define DECL_PROF __attribute__((cleanup(prof_func_end))) struct prof_entry_t prof_entry = {__FILE__, __func__, __LINE__, get_usec(), 0};

int test_profile()
{
  DECL_PROF;
  usleep(100000);
  return 0;
}

int main()
{
  return test_profile();
}
