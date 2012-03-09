#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>

#define mfence() __asm__("mfence");
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

struct Notify
{
  volatile int64_t req_seq_;
  volatile int64_t ack_seq_;
  int req(){
    req_seq_++;
  }
  int ack(){
    ack_seq_ = req_seq_;
  }
  bool sync(){
    return ack_seq_ == req_seq_;
  }
};

enum {
  RUN,
  STANDBY,
};
int64_t n = 0;
int64_t counter = 0;
volatile int64_t state = STANDBY;
Notify notify;
void* do_work(void* arg)
{
  for(int64_t i = 0; i < n;){
    notify.ack();
    switch(state)
    {
      case RUN:
        counter++;
        i++;
        break;
      case STANDBY:
        break;
    }
  }
}

int64_t test_without_notify()
{
  int n_err = 0;
  pthread_t thread;
  state = RUN;
  counter = 0;
  pthread_create(&thread, NULL, do_work, NULL);
  for(int64_t i = 0; i < n; i++){
    state = STANDBY;
    counter++;
    state = RUN;
  }
  pthread_join(thread, NULL);
  return 2 * n - counter;
}

int64_t test_with_notify()
{
  int n_err = 0;
  pthread_t thread;
  state = RUN;
  counter = 0;
  pthread_create(&thread, NULL, do_work, NULL);
  for(int64_t i = 0; i < n; i++){
    state = STANDBY;
    notify.req();
    while(!notify.sync())
      ;
    counter++;
    state = RUN;
  }
  pthread_join(thread, NULL);
  return 2 * n - counter;
}

int main(int argc, char *argv[])
{
  int i = (argc > 1)? atoi(argv[1]): 0;
  n = 1<<(i?:10);
  printf("n=%ld\n", n);
  profile(test_without_notify());
  profile(test_with_notify());
  return 0;
}
