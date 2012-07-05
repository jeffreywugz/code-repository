#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <signal.h>

typedef void (*sa_handler_t)(int);
int set_sigaction(int sig_no, sa_handler_t int_handler)
{
  struct sigaction new_action, old_action;
  new_action.sa_handler = int_handler;
  sigemptyset(&new_action.sa_mask);
  new_action.sa_flags = 0;
  return sigaction(sig_no, &new_action, &old_action);
};

void int_handler(int sig_no)
{
  return;
}

#define pause()
int test_madvise_noneed(int64_t len)
{
  int err = 0;
  const char* path = "file.mmap";
  int fd = 0;
  char* buf = NULL;
  assert(0 == set_sigaction(SIGINT, int_handler));
  assert(0 <= (fd = open(path, O_CREAT|O_RDWR, S_IRWXU)));
  assert(fd > 0);
  assert(0 == ftruncate(fd, len));
  printf("test madvise[len=%ld], C-c to mmap()\n", len);
  system("free -m|head -2");
  pause();
  assert(NULL != (buf = mmap(NULL, len, PROT_WRITE|PROT_READ, MAP_PRIVATE, fd, 0)));
  printf("mmap() OK, C-c to memset()\n");
  system("free -m|head -2");
  pause();
  memset(buf, 0, len);
  printf("memset() OK, C-c to madvise()\n");
  system("free -m|head -2");
  pause();
  memset(buf, len, MADV_DONTNEED);
  printf("madvise() OK, C-c to return\n");
  system("free -m|head -2");
  pause();
  return err;
}

int main(int argc, char *argv[])
{
  int err = 0;
  if (argc != 2)
  {
    fprintf(stderr, "%s size", argv[0]);
  }
  else
  {
    err = test_madvise_noneed(atoll(argv[1]));
  }
  return err;
}
