#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

int panic(const char* msg)
{
  perror(msg);
  exit(-1);
}

void test_process_limit(int64_t n)
{
  for(int64_t i = 0; i < n; i++) {
    fprintf(stderr, "fork i=%ld: total=%ld\n", i, 1<<i);
    if(fork() < 0)panic("fork:");
  }
}

int main(int argc, char *argv[])
{
  int i = (argc > 1)? atoi(argv[1]): 0;
  i = (i == 0)? 3: i;
  test_process_limit(i);
  return 0;
}
