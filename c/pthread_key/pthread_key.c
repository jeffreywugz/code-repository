#include <stdio.h>
#include <pthread.h>

int pthread_key_create(pthread_key_t *key, void (*destructor)(void*));

void on_exit(void* ptr)
{
  printf("destructor:%ld\n", pthread_self());
}
pthread_key_t key;

void* start_thread(void* p)
{
  pthread_setspecific(key, (void*)2);
  printf("in thread: td=%ld\n", pthread_self());
}
int main()
{
  pthread_t thread;
  pthread_key_create(&key, on_exit);
  pthread_create(&thread, NULL, start_thread, NULL);
  pthread_join(thread, NULL);
  pthread_setspecific(key, (void*)2);
  return 0;
}
