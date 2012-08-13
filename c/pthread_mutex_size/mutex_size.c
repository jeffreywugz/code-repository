#include <pthread.h>

int main()
{
  printf("sizeof(mutex_t)=%ld\n", sizeof(pthread_mutex_t));
  return 0;
}
