#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int64_t get_file_limit(int64_t n)
{
  int64_t i = 0;
  int fd = 0;
  char path[1<<8];
  const char* dir = "limit_test_dir";
  snprintf(path, sizeof(path), "rm -rf %s", dir);
  assert(0 == system(path));
  assert(0 <= mkdir(dir,  ~0));
  for(i = 0; fd >= 0 && i < n; i++){
    snprintf(path, sizeof(path), "%s/%ld", dir, i);
    fd = open(path, O_CREAT);
    printf("open('%s')=>%d\n", path, fd);
    if (fd < 0)perror("open:");
  }
  return i;
}

int main()
{
  printf("opened fd number limit: %ld\n", get_file_limit(1<<20));
  return 0;
}
