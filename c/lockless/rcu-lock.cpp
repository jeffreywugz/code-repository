#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
#define ESTATE 1042
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
      int err = 0;
      if (NULL == items || 0 >= capacity)
      {
        err = -EINVAL;
      }
      else
      {
        items_ = items;
        capacity_ = capacity;
      }
      return err;
    }

    int push(void* p) {
      int64_t rear = rear_;
      if (NULL == items_ || rear&1 || (front_>>1) + capacity_ <= (rear>>1) || !__sync_bool_compare_and_swap(&rear_, rear, rear+1))
        return -EAGAIN;
      items_[(rear>>1)%capacity_] = p;
      rear_++;
      return 0;
    }

    int pop(void*& p) {
      int64_t front = front_;
      if (NULL == items_ || front&1 || front+1 >= rear_ || !__sync_bool_compare_and_swap(&front_, front, front+1))
        return -EAGAIN;
      p = items_[(front>>1)%capacity_];
      front_++;
      return 0;
    }
  private:
    int64_t capacity_;
    void** items_;
    volatile int64_t front_;
    volatile int64_t rear_;
};

struct ref_t
{
  volatile int64_t ref_;
  Queue* free_list_;
  // failed if return ref <= 1
  int64_t inc_ref() {
    return __sync_add_and_fetch(&ref_, 1);
  }
  int64_t dec_ref() {
    int err = 0;
    int64_t ref = __sync_add_and_fetch(&ref_, -1);
    while(0 == ref && NULL != free_list_ && -EAGAIN == (err = free_list_->push(this)))
      ;
    return ref;
  }
  ref_t* alloc() {
    int err = 0;
    ref_t* p = NULL;
    if (NULL != free_list_ && 0 == (err = free_list_->pop((void*&)p)))
    {
      p->ref_ = 1;
    }
    return p;
  }
};

ref_t* init_ref(Queue* queue, ref_t** items, char* buf, int64_t n_items, int64_t item_size) {
  int err = 0;
  ref_t* ref;
  if (NULL == queue || NULL == items || NULL == buf || 0 >= item_size || 0 >= n_items)
  {}
  else if (0 != (err = queue->init((void**)items, n_items)))
  {}
  for(int64_t i = 0; 0 == err && i < n_items; i++)
  {
    ((ref_t*)(buf + i*item_size))->ref_ = 0;
    ((ref_t*)(buf + i*item_size))->free_list_ = queue;
    err = queue->push(buf + i*item_size);
  }
  if (0 == err && NULL != queue && 0 == (err = queue->pop((void*&)ref)))
  {
    ref->ref_ = 1;
  }
  return ref;
}

struct cur_ref_t
{
  cur_ref_t(): cur_(NULL) {}
  ~cur_ref_t() {}
  ref_t* volatile cur_;

  ref_t* get() {
    ref_t* cur = (ref_t*)cur_;
    if (NULL != cur && 1 >= cur->inc_ref())
      cur = NULL;
    return cur;
  }

  ref_t* commit(ref_t* ref) {
    ref_t* old_ref = cur_;
    cur_ = ref;
    if (NULL != old_ref)old_ref->dec_ref();
    return cur_;
  }
};

struct Node : public ref_t
{
  int64_t val_;
};

ref_t* gref;
cur_ref_t cur_ref;
int64_t do_read(int64_t n)
{
  int64_t n_err = 0;
  ref_t* ref = NULL;
  for(int64_t i = 0; i < n; i++){
    ref = cur_ref.get();
    if (NULL == ref)
      n_err++;
  }
  return n_err;
}

int64_t do_update(int64_t n)
{
  int64_t n_err = 0;
  ref_t* old_ref = NULL;
  ref_t* new_ref = NULL;
  for(int64_t i = 0; i < n; i++){
    old_ref = cur_ref.get();
    if (NULL == old_ref){
      n_err++;
      continue;
    }
    new_ref = old_ref->alloc();
    cur_ref.commit(new_ref);
  }
  return n_err;
}

typedef void*(*pthread_handler_t)(void*);
int64_t test_rcu_lock(int64_t n)
{
  int64_t n_err = 0;
  int64_t total_err = 0;
  Queue queue;
  pthread_t thread[1<<1];
  char* buf = (char*)malloc(n * sizeof(void*) + n * sizeof(Node));
  assert(buf);
  gref = init_ref(&queue, (ref_t**)buf, buf + n*sizeof(void*), n, sizeof(Node));
  assert(gref);
  cur_ref.cur_ = gref;
  for(int64_t i = 0; i < array_len(thread); i++)
    pthread_create(thread + i, NULL, (pthread_handler_t)(i&1 ? do_read: do_update), (void*)n);
  for(int64_t i = 0; i < array_len(thread); i++){
    pthread_join(thread[i], (void**)&n_err);
    total_err += n_err;
  }
  return total_err;
}

int main(int argc, char *argv[])
{
  int i = (argc > 1)? atoi(argv[1]): 0;
  int64_t n = 1<<(i?:10);
  printf("n=%ld\n", n);
  profile(test_rcu_lock(n));
  return 0;
}
