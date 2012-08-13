#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <new>

class Queue
{
  public:
    Queue(): capacity_(0), items_(NULL), front_(0), rear_(0) {}
    ~Queue() {}

    int init(int64_t capacity, void** items) {
      int err = 0;
      capacity_ = capacity;
      items_ = items;
      return err;
    }

    int push(void* p) {
      int64_t rear = rear_;
      if (NULL == items_ || rear&1 || front_+(capacity_<<1) <= rear+1 || !__sync_bool_compare_and_swap(&rear_, rear, rear+1))
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

class MultiQueue
{
  public:
    MultiQueue(): n_queue_(0), queues_(NULL) {}
    ~MultiQueue() {}
    int init(int64_t n_queue, Queue* queues) {
      int err = 0;
      n_queue_ = n_queue;
      queues_ = queues;
      return err;
    }
    int push(int64_t idx, void* p) {
      return n_queue_ > 0? queues_[idx % n_queue_].push(p): -EAGAIN;
    }
    int pop(int64_t idx, void*& p) {
      return n_queue_ > 0? queues_[idx % n_queue_].pop(p): -EAGAIN;
    }
  private:
    int64_t n_queue_;
    Queue* queues_;
};

MultiQueue* new_mq(int64_t n_queue, int64_t queue_capacity)
{
  int err = 0;
  MultiQueue* mq = NULL;
  Queue* queue = NULL;
  char* buf = NULL;
  char* p = NULL;
  fprintf(stderr, "new_mq(n_queue=%ld, queue_capacity=%ld)\n", n_queue, queue_capacity);
  if (NULL == (buf = (char*)malloc(sizeof(MultiQueue) + n_queue * sizeof(Queue) + n_queue * queue_capacity * sizeof(void*))))
  {
    err = -ENOMEM;
  }
  else
  {
    mq = (MultiQueue*)buf;
    queue = (Queue*)(mq + 1);
    p = (char*)(queue + n_queue);
  }
  for(int64_t i = 0; 0 == err && i < n_queue; i++)
  {
    err = (new(queue + i)Queue())->init(queue_capacity, (void**)(p + i*queue_capacity*sizeof(void*)));
  }
  if (0 == err)
  {
    (new(mq)MultiQueue())->init(n_queue, queue);
  }
  if (0 != err)
  {
    fprintf(stderr, "new_mq(n_queue=%ld, queue_capacity=%ld)=>%d\n", n_queue, queue_capacity, err);
    free(buf);
    buf = NULL;
    mq = NULL;
  }
  return mq;
}

int64_t n_thread = 0;
int64_t n_item = 0;
MultiQueue* mq = NULL;

int do_work(int64_t idx)
{
  int err = 0;
  int64_t n_push_wait = 0;
  int64_t n_pop_wait = 0;
  void* p = NULL;
  if (NULL == mq)
  {
    err = -EINVAL;
  }
  for(int64_t i = 0; 0 == err && i < n_item; i++)
  {
    if (0 == err && -EAGAIN == (err = mq->push(random()%n_thread, (void*)(idx * n_item + i))))
    {
      err = 0;
      n_push_wait++;
    }
    if (0 == err && -EAGAIN == (err = mq->pop(idx, p)))
    {
      err = 0;
      n_pop_wait++;
    }
  }
  fprintf(stderr, "worker[%ld] push_wait=%ld, pop_wait=%ld\n", idx, n_push_wait, n_pop_wait);
  return err;
}

typedef void*(*pthread_handler_t)(void*);
int test_mq(const int64_t n_queue, const int64_t queue_capacity, const int64_t n_thread, const int64_t n_item)
{
  int err = 0;
  pthread_t thread[n_thread];
  ::n_thread = n_thread;
  ::n_item = n_item;
  assert(mq = new_mq(n_queue, queue_capacity));
  for(int64_t i = 0; i < n_thread; i++)
    pthread_create(thread + i, NULL, (pthread_handler_t)(do_work), (void*)i);
  for(int64_t i = 0; i < n_thread; i++)
    pthread_join(thread[i], NULL);
  free(mq);
  return err;
}

int main(int argc, char *argv[])
{
  int err = 0;
  if (argc != 5)
  {
    err = -EINVAL;
    fprintf(stderr, "%s n_queue queue_capacity n_thread n_item", argv[0]);
  }
  else
  {
    err = test_mq(atoll(argv[1]), atoll(argv[2]), atoll(argv[3]), atoll(argv[4]));
  }
  return err;
}
