#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>

int64_t __stop = 0;

void sighand(int signo)
{
  __stop++;
}

void install_sighandler()
{
  int err = 0;
  struct sigaction actions;
  memset(&actions, 0, sizeof(actions));
  sigemptyset(&actions.sa_mask);
  actions.sa_flags = 0;
  actions.sa_handler = sighand;
  if (0 != sigaction(SIGINT, &actions, NULL))
  {
    fprintf(stderr, "sigaction()=>%s\n", strerror(errno));
  }
}

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

#define block_size (1<<9)
#define n_blocks (1<<12)
char file_block[block_size * n_blocks] __attribute__ ((aligned (block_size))) ;
struct exp_stat_t
{
  int64_t count[64];
  int64_t time[64];
};

int exp_stat_init(struct exp_stat_t* stat)
{
  memset(stat, 0, sizeof(*stat));
  return 0;
}

int64_t exp_get_idx(const int64_t x)
{
  return x? (64 - __builtin_clzl(x)): 0;
}

int exp_stat_add(struct exp_stat_t* stat, int64_t x)
{
  int64_t idx = exp_get_idx(x);
  stat->count[idx]++;
  stat->time[idx]+=x;
}

int exp_stat_report(struct exp_stat_t* stat)
{
  int64_t total_count = 0;
  int64_t total_time = 0;
  for(int64_t i = 0; i < 64; i++)
  {
    total_count += stat->count[i];
    total_time += stat->time[i];
  }
  printf("total_count=%'ld, total_time=%'ldus, avg=%'ldus\n", total_count, total_time, total_time/total_count);
  for(int64_t i = 0; i < 32; i++)
  {
    printf("stat[..<latency<2**%d]: %lf%%, count=%'ld, time=%'ldus, avg=%'ldus\n",
           i, 100.0*stat->count[i]/(total_count+1), stat->count[i], stat->time[i], stat->count[i] == 0? 0: stat->time[i]/stat->count[i]);
  }
  return 0;
}


void* cpu_loop(void* arg)
{
  int64_t i = 0;
  int64_t last_ts = 0;
  int64_t cur_ts = 0;
  int64_t delta_ts = 0;
  struct exp_stat_t stat;
  exp_stat_init(&stat);
  while(!__stop)
  {
    if ((int64_t)arg == 0)
    {
      for(i = 0; i < 1LL<<14; i++)
        ;
      cur_ts = get_usec();
      delta_ts = cur_ts - last_ts;
      if (last_ts > 0)
      {
        exp_stat_add(&stat, delta_ts);
      }
      last_ts = cur_ts;
    }
  }
  if ((int64_t)arg == 0)
  {
    exp_stat_report(&stat);
  }
  return NULL;
}

void* sleep_loop(void* arg)
{
  while(1)
    usleep(1000000);
  return NULL;
}

int get_iops(const char* path, int64_t total_read_ops)
{
  int err = 0;
  int fd = 0;
  int64_t size = 0;
  int64_t blk_idx = 0;
  int64_t start_time = 0;
  int64_t end_time = 0;
  printf("get_disk_iops(path=%s, total_read_ops=%ld)\n", path, total_read_ops);
  if (NULL == path || 0 >= total_read_ops)
  {
    err = -EINVAL;
  }
  else if (0 > (fd = open(path, O_RDONLY|O_DIRECT)))
  {
    err = errno;
    perror("open");
  }
  else if (0 != ioctl(fd, BLKGETSIZE64, &size))
  {
    err = errno;
    perror("ioctl");
  }
  else
  {
    printf("%s size=%ld\n", path, size);
  }
  start_time = get_usec();
  for(int64_t s = 0; 0 == err && s < total_read_ops; s++)
  {
    blk_idx = random() % (size/block_size);
    lseek(fd, blk_idx * block_size, SEEK_SET);
    if (0 >= read(fd, file_block, block_size))
    {
      err = errno;
      perror("read");
    }
  }
  end_time = get_usec();
  printf("read ops: %ld/%ldus = %.2fOPS/s\n", total_read_ops, end_time - start_time, 1e6 * total_read_ops/(end_time - start_time + 1));

  lseek(fd, 0, SEEK_SET);
  return err;
}

void* io_loop(void* arg)
{
  const int64_t n_read_ops = 1<<12;
  while(1)
    get_iops(getenv("disk_path")?: "/dev/sda", n_read_ops);
  return NULL;
}

typedef void*(*pthread_handler_t)(void*);
int gen_load(const char* type, const int64_t n_threads)
{
  pthread_t threads[n_threads];
  pthread_handler_t handler = NULL;
  if (0 == strcmp(type, "cpu")){
    handler = cpu_loop;
  } else if (0 == strcmp(type, "sleep")) {
    handler = sleep_loop;
  } else if (0 == strcmp(type, "disk")) {
    handler = io_loop;
  }
  for (int64_t i = 0; i < n_threads; i++){
    pthread_create(threads + i, NULL, (pthread_handler_t)handler, i);
  }
  for (int64_t i = 0; i < n_threads; i++){
    pthread_join(threads[i], NULL);
  }
  return 0;
}


int main(int argc, char *argv[])
{
  int err = 0;
  install_sighandler();
  if (argc != 3) {
    err = -EINVAL;
    fprintf(stderr, "Usages:\n" "\t./load-gen [sleep|cpu|disk] n_threads\n");
  } else {
    err = gen_load(argv[1], atoll(argv[2]));
  }
  return err;
}
