#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <new>

using namespace std;

template<typename T, typename ID>
struct TSIFactory
{
  public:
    TSIFactory(): create_err_(0) {
      create_err_ = pthread_key_create(&key_, destroy);
    }
    ~TSIFactory() {
      if (0 == create_err_)
        pthread_key_delete(key_);
    }
    T* _get(){
      int err = 0;
      T* val = NULL;
      if (create_err_ != 0)
      {}
      else if (NULL != (val = (T*)pthread_getspecific(key_)))
      {}
      else if (NULL == (val = new(std::nothrow)T()))
      {
        err = -ENOMEM;
      }
      else if (0 != (err = pthread_setspecific(key_, val)))
      {}
      if (0 != err && NULL != val)
      {
        destroy(val);
        val = NULL;
      }
      return val;
    }
    static T* get() {
      static TSIFactory<T, ID> factory;
      return factory._get();
    }
  private:
    static void destroy(void* arg) {
      if (NULL != arg)
      {
        delete (T*)arg;
      }
    }
    int create_err_;
    pthread_key_t key_;
};

class A
{
  public:
  A() { printf("A()\n"); }
  ~A() { printf("~A()\n"); }
};

class xx{};
void* foo(void* arg)
{
  A* a = TSIFactory<A, xx>::get();
  if (NULL == a)
  {
    printf("a == NULL\n");
  }
  printf("pthread[%ld] exit!\n", arg);
  return a;
}

int main(int argc, char *argv[])
{
  pthread_t thread1, thread2;
  pthread_create(&thread1, NULL, foo, (void*)1);
  pthread_create(&thread2, NULL, foo, (void*)2);
  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);
  return 0;
}
