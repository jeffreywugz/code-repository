#include <stdio.h>
#include <unistd.h>

int main()
{
  while(1) {
    printf("heartbeat from main.\n");
    usleep(1000000);
  }
  return 0;
}
