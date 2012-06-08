#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

int file_map_write(const char* path, const int64_t len, char** buf)
{
  int err = 0;
  int fd = 0;
  if (NULL == path || NULL == buf)
  {
    err = -EINVAL;
  }
  else if ((fd =open(path, O_RDWR|O_CREAT, S_IRWXU|S_IRGRP)) < 0)
  {
    err = -EIO;
    fprintf(stderr, "open(%s):%s\n", path, strerror(errno));
  }
  else if (0 != ftruncate(fd, len))
  {
    err = -EIO;
    fprintf(stderr, "ftruncate(%s):%s\n", path, strerror(errno));
  }
  else if (NULL == (*buf = (char*)mmap(NULL, len, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0)) || MAP_FAILED == buf)
  {
    buf = NULL;
    err = -EIO;
    fprintf(stderr, "mmap[%s] failed:%s\n", path, strerror(errno));
  }
  if (fd > 0)
    close(fd);
  return err;
}

int test_mmap_truncate(const char* path)
{
  int err = 0;
  char* buf = NULL;
  int64_t len = sizeof(int64_t);
  if (0 != (err = file_map_write(path, len, &buf)))
  {
    fprintf(stderr, "file_map_write(%s)=>%s\n", path, strerror(err));
  }
  else if (0 != truncate(path, 0))
  {
    fprintf(stderr, "truncate(%s)=>%s\n", path, strerror(errno));
  }
  else
  {
    printf("file[%s] truncate to 0\n", path);
    buf[0] = 0;
    printf("access buf[mapped %s] success\n");
  }
  return err;
}

int main(int argc, char *argv[])
{
  const char* path = "./file.map";
  return test_mmap_truncate(path);
}
