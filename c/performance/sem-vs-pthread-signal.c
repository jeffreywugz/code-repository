#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#include <semaphore.h>
#include <pthread.h>
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
  printf("%s=>%ld in %ldms\n", #expr, result, new_us - old_us);           \
  new_us - old_us; })

void gen_task() {}
void do_task() {}

int64_t n;
int64_t counter = 0;
sem_t* sem;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

void* consumer_sync_with_sem(void* arg)
{
  for (int64_t i = 0; i < n; i++) {
    sem_wait(sem);
    do_task();
    counter--;
  }
}

void* producer_sync_with_sem(void* arg)
{
  for(int64_t i = 0; i < n; i++) {
    gen_task();
    counter++;
    sem_post(sem);
  }
}

void* consumer_sync_with_pthread_cond(void* arg)
{
  for(int64_t i = 0; i < n; i++) {
    pthread_mutex_lock(&mutex);
    while(counter <= 0) {
      pthread_cond_wait(&cond, &mutex);
    }
    do_task();
    counter--;
    pthread_mutex_unlock(&mutex);
  }
}

void* producer_sync_with_pthread_cond(void* arg)
{
  for(int64_t i = 0; i < n; i++) {
    pthread_mutex_lock(&mutex);
    gen_task();
    counter++;
    if (counter == 1)
      pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mutex);
  }
}

int64_t test_sem()
{
  const char* file_name = "file.sem";
  int value = 1;
  pthread_t thread1;
  pthread_t thread2;
  sem = sem_open(file_name, O_CREAT, S_IRWXU, value);
  assert(sem != SEM_FAILED);
  sem_unlink(file_name);
  pthread_create(&thread1, NULL, consumer_sync_with_sem, NULL);
  pthread_create(&thread2, NULL, producer_sync_with_sem, NULL);

  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);
  sem_close(sem);
  return 0;
}

int64_t test_pthread_cond()
{
  pthread_t thread1;
  pthread_t thread2;
  pthread_create(&thread1, NULL, consumer_sync_with_pthread_cond, NULL);
  pthread_create(&thread2, NULL, producer_sync_with_pthread_cond, NULL);

  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);
  return 0;
}

int main(int argc, char *argv[])
{
  n = argc > 1 ? atoi(argv[1]): 10;
  n = (1 << (n > 0 ? n: 10));
  printf("n=%ld\n", n);
  profile(test_sem());
  profile(test_pthread_cond());
  return 0;
}
