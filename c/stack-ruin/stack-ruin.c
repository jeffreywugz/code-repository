#define __noinline__ __attribute__((noinline))
int __noinline__ core(){ return *((char*)0) = 0; }
int __noinline__ foo4() {return core(); }
int __noinline__ foo3() {return foo4(); }
int __noinline__ foo2() {return foo3(); }
int __noinline__ foo1() {return foo2(); }

int stackoverflow(int x){ char buf[0]; memset(buf, 0, 16); return 0; }
int __noinline__ foo(int x) { return x > 0? stackoverflow(x): foo(x) * 2; }

#include <pthread.h>
typedef void*(*pthread_handler_t)(void*);
pthread_mutex_t lock1;
pthread_mutex_t lock2;
int worker1(void* arg)
{
  pthread_mutex_lock(&lock1);  sleep(1); pthread_mutex_lock(&lock2);
  return 0;
}
int worker2(void* arg)
{
  pthread_mutex_lock(&lock2);  sleep(1); pthread_mutex_lock(&lock1);
  return 0;
}

int start_worker()
{
  pthread_t thread1;
  pthread_t thread2;
  pthread_create(&thread1, NULL, (pthread_handler_t)worker1, NULL);
  pthread_create(&thread2, NULL, (pthread_handler_t)worker2, NULL);
  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);
  return 0;
}
int main()
{
  return start_worker();
}
