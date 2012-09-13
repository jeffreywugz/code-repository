#include <errno.h>
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

#define profile(expr, n) { \
  int64_t start = get_usec(); \
  expr;\
  int64_t end = get_usec();\
  printf("%s: 1000000*%ld/%ld=%ld\n", #expr, n, end - start, 1000000 * n / (end - start)); \
}

void gen_task() {}
void do_task() {}

int64_t n;
volatile int64_t counter = 0;
sem_t* sem;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
int set_affinity(pthread_t thread, int k)
{
  int err = 0;
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(k, &cpuset);
  err = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (err != 0)
    perror(errno);
  return err;
}

void* consumer_sync_with_sem(void* arg)
{
  set_affinity(pthread_self(), (int)(arg));
  for (int64_t i = 0; i < n; i++) {
    sem_wait(sem);
    do_task();
    counter--;
  }
}

void* producer_sync_with_sem(void* arg)
{
  set_affinity(pthread_self(), (int)(arg));
  for(int64_t i = 0; i < n; i++) {
    gen_task();
    __sync_add_and_fetch(&counter, 1);
    sem_post(sem);
  }
}

void* consumer_sync_with_pthread_cond(void* arg)
{
  set_affinity(pthread_self(), (int)(arg));
  for(int64_t i = 0; i < n; i++) {
    pthread_mutex_lock(&mutex);
    while(counter <= 0) {
      pthread_cond_wait(&cond, &mutex);
    }
    do_task();
    __sync_add_and_fetch(&counter, -1);
    pthread_mutex_unlock(&mutex);
  }
}

void* producer_sync_with_pthread_cond(void* arg)
{
  set_affinity(pthread_self(), (int)(arg));
  for(int64_t i = 0; i < n; i++) {
    pthread_mutex_lock(&mutex);
    gen_task();
    __sync_add_and_fetch(&counter, 1);
    //if (counter == 1)
      pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mutex);
  }
}

sem_t* get_sem(int shared)
{
  sem_t* sem = NULL;
  static sem_t static_sem;
  const char* file_name = "file.sem";
  if (shared) {
    sem = sem_open(file_name, O_CREAT, S_IRWXU, 0);
    assert(sem != SEM_FAILED);
    sem_unlink(file_name);
  } else {
    sem = &static_sem;
    assert(0 == sem_init(sem, 0, 0));
  }
  return sem;
}

int64_t test_sem(const int64_t n, int shared)
{
  pthread_t thread1[n];
  pthread_t thread2[n];
  sem = get_sem(shared);
  counter = 0;
  for(int64_t i = 0; i < n; i++){
    pthread_create(&thread1[i], NULL, consumer_sync_with_sem, i);
    pthread_create(&thread2[i], NULL, producer_sync_with_sem, n + i);
  }

  for(int64_t i = 0; i < n; i++){
    pthread_join(thread1[i], NULL);
    pthread_join(thread2[i], NULL);
  }
  sem_close(sem);
  return 0;
}

int64_t test_pthread_cond(const int64_t n)
{
  pthread_t thread1[n];
  pthread_t thread2[n];
  counter = 0;
  for(int64_t i = 0; i < n; i++){
    pthread_create(&thread1[i], NULL, consumer_sync_with_pthread_cond, i);
    pthread_create(&thread2[i], NULL, producer_sync_with_pthread_cond, n + i);
  }

  for(int64_t i = 0; i < n; i++) {
    pthread_join(thread1[i], NULL);
    pthread_join(thread2[i], NULL);
  }
  return 0;
}

int main(int argc, char *argv[])
{
  int n_thread = 0;
  n = argc > 1 ? atoi(argv[1]): 10;
  n = (1 << (n > 0 ? n: 10));
  n_thread = argc > 2 ? atoi(argv[2]): 1;
  printf("n=%ld\n", n);
  profile(test_sem(n_thread, 0), n);
  profile(test_sem(n_thread, 1), n);
  profile(test_pthread_cond(n_thread), n);
  return 0;
}
