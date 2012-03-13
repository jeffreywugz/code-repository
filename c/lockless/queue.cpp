#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <errno.h>
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


class Queue
{
  public:
    Queue(): capacity_(0), items_(NULL), front_(0), rear_(0) {}
    ~Queue() {}

    int init(void** items, int64_t capacity) {
      items_ = items;
      capacity_ = capacity;
    }

    int push(void* p) {
      int64_t rear = rear_;
      if (NULL == items_ || rear&1 || front_+(capacity_<<1) <= rear+1 || !__sync_bool_compare_and_swap(&rear_, rear, rear+1))
        return -EAGAIN;
      items_[rear++>>1] = p;
      return 0;
    }

    int pop(void*& p) {
      int64_t front = front_;
      if (NULL == items_ || front&1 || front+1 >= rear_ || !__sync_bool_compare_and_swap(&front_, front, front+1))
        return -EAGAIN;
      p = items_[front_++>>1];
      return 0;
    }
  private:
    int64_t capacity_;
    void** items_;
    volatile int64_t front_;
    volatile int64_t rear_;
};

Queue queue;
int64_t do_pop(int64_t n)
{
  void* p = NULL;
  int64_t n_confilict = 0;
  for(int64_t i = 0; i < n; i++) {
    if (0 != queue.pop(p))
      n_confilict++;
  }
  return n_confilict;
}

int64_t do_push(int64_t n)
{
  void* p = NULL;
  int64_t n_confilict = 0;
  for(int64_t i = 0; i < n; i++) {
    if (0 != queue.push(p))
      n_confilict++;
  }
  return n_confilict;
}

typedef void*(*pthread_handler_t)(void*);
int64_t test_queue(int64_t n)
{
  int64_t total_confilict = 0;
  int64_t n_confilict = 0;
  pthread_t thread[1<<2];
  void** buf = (void**)malloc(n * sizeof(void*));
  assert(buf);
  queue.init(buf, n);
  for(int64_t i = 0; i < array_len(thread); i++)
    pthread_create(thread+i, NULL, (pthread_handler_t)(i&1? do_pop: do_push), (void*)n);
  for(int64_t i = 0; i < array_len(thread); i++){
    pthread_join(thread[i], (void**)&n_confilict);
    total_confilict += n_confilict;
  }
  if(buf)free(buf);
  return total_confilict;
}

int main(int argc, char *argv[])
{
  int64_t n = (argc > 1)? atoi(argv[1]): 0;
  n = 1<<(n?:10);
  printf("n=%ld\n", n);
  profile(test_queue(n));
  return 0;
}
