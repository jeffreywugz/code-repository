#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>

#define array_len(x) (sizeof(x)/sizeof(x[0]))
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
  printf("%s=>%ld in %ldms\n", #expr, result, new_us - old_us);           \
  new_us - old_us; })

struct
{
  Stack(): next_top_(0), top_(0)
  int64_t next_top_;
  int64_t top_;
  void** items_;
  int pop(void*& p)
  {
  }
  int push(void* p)
  {
  }
};

struct SlicePool
{
  int64_t ref_
};

TicketLock ticket_lock;
int64_t s = 0;
int64_t n = 0;
int64_t do_work_with_lock(int64_t idx)
{
  int64_t n_err = 0;
  for(int64_t i = 0; i < n; i++){
    int64_t ticket = ticket_lock.get_ticket();
    while(!ticket_lock.wait(ticket))
      ;
    s++;
    if (s > 1)
      n_err++;
    s--;
    ticket_lock.done(ticket);
  }
  return n_err;
}

int64_t do_work_no_lock(int64_t idx)
{
  int64_t n_err = 0;
  for(int64_t i = 0; i < n; i++){
    s++;
    if (s >= 2)
      n_err++;
    s--;
  }
  return n_err;
}

enum {USE_LOCK, NO_LOCK};
typedef void*(*pthread_handler_t)(void*);
int64_t test_ticket_lock(int flag)
{
  int64_t n_err = 0;
  int64_t total_err = 0;
  pthread_t thread[1<<1];
  s = 0;
  for(int64_t i = 0; i < array_len(thread); i++)
    pthread_create(thread + i, NULL, (pthread_handler_t)(flag == USE_LOCK? do_work_with_lock: do_work_no_lock), (void*)i);
  for(int64_t i = 0; i < array_len(thread); i++){
    pthread_join(thread[i], (void**)&n_err);
    total_err += n_err;
  }
  return n_err;
}

int main(int argc, char *argv[])
{
  int i = (argc > 1)? atoi(argv[1]): 0;
  n = 1<<(i?:10);
  printf("n=%ld\n", n);
  profile(test_ticket_lock(NO_LOCK));
  profile(test_ticket_lock(USE_LOCK));
  return 0;
}
