#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>

int main()
{
  printf("sizeof(int)=%lu, sizeof(long)=%ld, sizeof(void*)=%ld, sizeof(long long)=%ld\n",
         sizeof(int), sizeof(long), sizeof(void*), sizeof(long long));
  printf("sizeof(pthread)=%lu, sizeof(pthread_mutex)=%lu, sizeof(pthread_cond)=%lu, sizeof(sem_t)=%lu\n",
         sizeof(pthread_t), sizeof(pthread_mutex_t), sizeof(pthread_cond_t), sizeof(sem_t));
  return 0;
}
