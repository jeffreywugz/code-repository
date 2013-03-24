#include <errno.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/time.h>

const char* usages = "scan_type=scandir|readdir|getdents %$1s dir-path scan-times";
#define Log(level, fmt...) { fprintf(stderr, #level " " fmt); fprintf(stderr, "\n"); }

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

static int is_clog_file(const struct dirent* a)
{
  return atoll(a->d_name) > 0;
}

static char* select_first_file_from_dir_by_scan(char* buf, int64_t limit, const char* dir)
{
  int err = 0;
  char* first = NULL;
  struct dirent **list = NULL;
  int n = 0;
  int64_t len = 0;
  if (NULL == buf || limit <= 0 || NULL == dir)
  {
    err = -EINVAL;
    Log(ERROR, "invalid argument, buf=%p[%ld], dir=%s", buf, limit, dir);
  }
  else if ((n = scandir(dir, &list, is_clog_file, versionsort)) < 0)
  {
    Log(ERROR, "scandir(%s)=>%s", dir, strerror(errno));
  }
  else if (0 == n)
  {
    Log(WARN, "%s is empty, can not select a file", dir);
  }
  else if ((len = snprintf(buf, limit, "%s/%s", dir, (*list)->d_name)) <= 0
           || len >= limit)
  {
    Log(ERROR, "generate first_file_name error, buf=%p[%ld], dir=%s, first=%s",
              buf, limit, dir, first);
  }
  else
  {
    first = buf;
  }
  if (NULL == list && n > 0)
  {
    while(n--)
    {
      free(list[n]);
    }
    free(list);
  }
  return first;
}


static char* select_first_file_from_dir(char* buf, int64_t limit, const char* dir_name)
{
  char* first = NULL;
  int64_t len = 0;
  DIR* dir = NULL;
  struct dirent* ent = NULL;
  if (NULL == (dir = opendir(dir_name)))
  {
    Log(ERROR, "opendir(%s) FAIL", dir_name);
  }
  else if (NULL == (ent = readdir(dir)))
  {
    Log(ERROR, "readdir(%s) FAIL", dir_name);
  }
  else if ((len = snprintf(buf, limit, "%s/%s", dir_name, ent->d_name)) < 0
           || len >= limit)
  {
    Log(ERROR, "gen file_name fail(dir=%s, file=%s)", dir_name, ent->d_name);
  }
  else
  {
    first = buf;
  }
  if (NULL != dir)
  {
    closedir(dir);
  }
  return first;
}

struct linux_dirent {
  long           d_ino;
  off_t          d_off;
  unsigned short d_reclen;
  char           d_name[];
};

static char* select_first_file_from_dir_by_getdents(char* buf, int64_t limit, const char* dir_name)
{
  char* first = NULL;
  int64_t nread = 0;
  int fd = -1;
  char entbuf[1024];
  struct linux_dirent *ent = NULL;

  if ((fd = open(dir_name, O_RDONLY | O_DIRECTORY)) < 0)
  {
    Log(ERROR, "opendir(%s) FAIL, %s", dir_name, strerror(errno));
  }
  else if ((nread = syscall(SYS_getdents, fd, entbuf, sizeof(entbuf))) < 0)
  {
    Log(ERROR, "getdents(%s) FAIL, %s", dir_name, dir_name);
  }
  else if (0 == nread)
  {
    Log(INFO, "dir %s is empty", dir_name);
  }
  for (int64_t pos = 0; pos < nread; pos += ent->d_reclen)
  {
    ent = (struct linux_dirent*)(entbuf + pos);
    char d_type = *(entbuf + pos + ent->d_reclen - 1);
    int64_t len = 0;
    if (d_type != DT_REG)
    {
      fprintf(stderr, "skip %s: not regular file\n", ent->d_name);
    }
    else if (atoll(ent->d_name) <= 0)
    {
      fprintf(stderr, "skip %s\n", ent->d_name);
    }
    else if ((len = snprintf(buf, limit, "%s/%s", dir_name, ent->d_name)) < 0
             || len >= limit)
    {
      Log(ERROR, "gen file_name fail(dir=%s, file=%s)", dir_name, ent->d_name);
      break;
    }
    else
    {
      first = buf;
      break;
    }
  }
  if (fd >= 0)
  {
    close(fd);
  }
  return first;
}

int test_scan_dir(const char* dir, int64_t n_loop)
{
  int err = 0;
  char file_name[128] = "None";
  const char* scan_type = getenv("scan_type")?: "scandir";
  fprintf(stderr, "test_scan_dir(dir=%s, n_loop=%ld, scan_type=%s)\n", dir, n_loop, scan_type);
  if (NULL == dir || n_loop < 0)
  {
    err = -EINVAL;
  }
  for(int64_t i = 0; 0 == err && i < n_loop; i++)
  {
    int64_t start_ts = 0, end_ts = 0;
    strcpy(file_name, "None");
    start_ts = get_usec();
    if (0 == strcmp(scan_type, "scandir"))
    {
      select_first_file_from_dir_by_scan(file_name, sizeof(file_name), dir);
    }
    else if (0 == strcmp(scan_type, "readdir"))
    {
      select_first_file_from_dir(file_name, sizeof(file_name), dir);
    }
    else if (0 == strcmp(scan_type, "getdents"))
    {
      select_first_file_from_dir_by_getdents(file_name, sizeof(file_name), dir);
    }
    end_ts = get_usec();
    Log(INFO, "select(%s)=>%s, %ldus",  dir, file_name, end_ts - start_ts);
  }
  return err;
}

int main(int argc, char** argv)
{
  int err = 0;
  if (argc != 3)
  {
    err = -EINVAL;
    fprintf(usages, argv[0]);
  }
  else if (0 != (err = test_scan_dir(argv[1], atoll(argv[2]))))
  {
    Log(ERROR, "test_scan_dir(%s, %s)=>%d", argv[1], argv[2], err);
  }
  return err;
}
