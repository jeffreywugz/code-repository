#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <locale.h>

int read_file(const char* file, int64_t offset, int64_t len)
{
  int err = 0;
  int fd = 0;
  int64_t read_count = 0;
  char buf[512];
  if (0 > (fd = open(file, O_RDWR|O_CREAT|O_DIRECT, S_IRWXU)))
  {
    err = errno;
    fprintf(stderr, "open(%s)=>%s\n", file, strerror(errno));
  }
  else if ((read_count = pread(fd, buf, len, offset)) < 0)
  {
    err = errno;
    fprintf(stderr, "pread(%s)=>%s\n", file, strerror(errno));
  }
  else
  {
    fprintf(stdout, "pread(file=%s, offset=%ld), read %ld bytes\n", file, offset, read_count);
  }
  if (fd > 0)
  {
    close(fd);
  }
  return err;
}

int main(int argc, char** argv)
{
  int err = 0;
  if (argc != 3)
  {
    err = -EINVAL;
  }
  else if (0 != (err = read_file(argv[1], atoll(argv[2]), 512)))
  {
    fprintf(stderr, "read_file(%s, %s)=>%d\n", argv[1], argv[2], err);
  }
  return err;
}
