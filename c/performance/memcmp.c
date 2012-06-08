#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/times.h>

void* list[1024 * 1024];

int main(void)
{
  int count = sizeof(list) / sizeof(char*);
  int i;
  for (i=0; i < count; i++)
    list[i] = calloc(1024, 1);

  int dupes = 0;
  int start = times(NULL);
  for (i=0; i<count-1; i++)
    if (!memcmp(list[i], list[i+1], 1024))
      dupes++;

  int ticks = times(NULL) - start;
  printf("Time: %d ticks (%d memcmp/tick)\n", ticks, dupes/ticks);

  return 0;
}
