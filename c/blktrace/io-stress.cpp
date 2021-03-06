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
#include <libaio.h>

const char* _usages = "Usages:\n"
  "path=tmp time_limit=3000000 write_size=1024 %1$s async|sync\n";
#define cfg(key, default_value) (getenv(key)?:default_value)
#define cfgi(key, default_value) atoll(cfg(key, default_value))
#define error(format,...) fprintf(stderr, "ERROR: %s:%d " format "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#define info(format,...) fprintf(stderr, "INFO: %s:%d " format "\n", __FILE__, __LINE__, ##__VA_ARGS__)

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

struct ExpStat
{
  ExpStat() { memset(this, 0, sizeof(*this)); }
  static int64_t get_idx(const int64_t x){ return x? (64 - __builtin_clzl(x)): 0; }
  int add(int64_t x)
  {
    int64_t idx = get_idx(x);
    count[idx]++;
    time[idx]+=x;
  }

  int report()
  {
    int64_t total_count = 0;
    int64_t total_time = 0;
    for(int64_t i = 0; i < 64; i++)
    {
      total_count += count[i];
      total_time += time[i];
    }
    printf("total_count=%'ld, total_time=%'ldus, avg=%'ldus\n", total_count, total_time, total_time/total_count);
    for(int64_t i = 0; i < 32; i++)
    {
      printf("stat[..<latency<2**%d]: %lf%%, count=%'ld, time=%'ldus, avg=%'ldus\n",
             i, 100.0*count[i]/(total_count+1), count[i], time[i], count[i] == 0? 0: time[i]/count[i]);
    }
    return 0;
  }

  int64_t count[64];
  int64_t time[64];
};


struct IOStat
{
  IOStat(): start_time_(0), end_time_(0), latency_(), total_write_bytes_(0), total_write_count_(0), total_write_time_(0) {}
  ~IOStat(){}
  void start() { start_time_ = get_usec(); }
  void end() { end_time_ = get_usec(); }
  void add(int64_t write_size, int64_t write_time){
    __sync_fetch_and_add(&total_write_count_, 1);
    __sync_fetch_and_add(&total_write_bytes_, write_size);
    __sync_fetch_and_add(&total_write_time_, write_time);
    latency_.add(write_time);
  }
  void report() {
    int64_t duration = end_time_ - start_time_;
    fprintf(stderr, "duration=%lds, BW=%lfM/s, IOPS=%ldIO/s\n",
            duration/1000000, 1.0 * total_write_bytes_/duration, 1000000 * total_write_count_/duration);
    latency_.report();
  }
  int64_t start_time_;
  int64_t end_time_;
  ExpStat latency_;
  int64_t total_write_bytes_;
  int64_t total_write_count_;
  int64_t total_write_time_;
};

class IOWorker
{
  public:
    IOWorker(const char* path, int64_t n_file, int64_t file_size): path_(path), n_file_(n_file), file_size_(file_size),
                                                                   fd_(-1), last_file_no_(-1), last_submit_time_(0){}
    virtual ~IOWorker() {}
    IOStat& get_stat(){ return stat_; }
    int64_t get_last_submit_time() { return last_submit_time_; }
    int submit(int64_t pos, char* buf, int64_t len) {
      int err = 0;
      int64_t file_no = (pos/file_size_) % n_file_;
      int fd = -1;
      int64_t cur_ts = get_usec();
      last_submit_time_ = cur_ts;
      if (pos < 0 || NULL == buf || len < 0)
      {
        err = -EINVAL;
      }
      else if (0 > (fd = get_fd(pos)))
      {
        error("get_fd(%s) faile", pos);
      }
      else if (0 != do_io(fd, buf, len, pos % file_size_, cur_ts))
      {
        error("do_io(pos=%ld) fail", pos);
      }
      return err;
    }

    virtual int do_io(int fd, const char* buf, int64_t len, int64_t offset, int64_t start_ts) {
      usleep(100);
       return 0;
    }
    int get_fd(int64_t pos){
      int err = 0;
      int64_t file_no = (pos/file_size_) % n_file_;
      char file_path[1024];
      if (last_file_no_ == file_no)
      {}
      else if (0 >= snprintf(file_path, sizeof(file_path), "%s/%ld", path_, file_no%n_file_))
      {
        err = -ENOMEM;
      }
      else
      {
        if (fd_ > 0)
        {
          close(fd_);
          fd_ = -1;
          last_file_no_ = -1;
        }
        if (0 > (fd_ = open(file_path, O_RDWR|O_CREAT|O_DIRECT, S_IRWXU)))
        {
          error("open(%s)=>%s", file_path, strerror(errno));
        }
        else
        {
          last_file_no_ = file_no;
          if ((file_no % 10) == 0)
          {
            info("file_no=%ld, pass=%ld", file_no, (pos/file_size_)/n_file_);
          }
        }
      }
      return fd_;
    }
  protected:
    IOStat stat_;
    const char* path_;
    int64_t n_file_;
    int64_t file_size_;
    int64_t fd_;
    int64_t last_file_no_;
    int64_t last_submit_time_;
};

class SyncIOWorker: public IOWorker
{
  public:
    SyncIOWorker(const char* path, int64_t n_file, int64_t file_size): IOWorker(path, n_file, file_size) {}
    virtual ~SyncIOWorker() {}
    int do_io(int fd, const char* buf, int64_t len, int64_t offset, int64_t start_ts) {
      int err = 0;
       if (len != pwrite(fd, buf, len, offset))
       {
         err = -EIO;
       }
       else if (0 != fsync(fd))
       {
         err = -EIO;
       }
       else
       {
         stat_.add(len, get_usec() - start_ts);
       }
       return err;
    }
};

class AsyncIOWorker: public IOWorker
{
  public:
    AsyncIOWorker(const char* path, int64_t n_file, int64_t file_size): IOWorker(path, n_file, file_size) {
      memset(&ctx_, 0, sizeof(ctx_));
      if(0 != io_setup(10, &ctx_))
      {
        error("io_setup error %s", strerror(errno));
      }
    }
    virtual ~AsyncIOWorker(){
      io_destroy(ctx_);
    }
    
    int do_io(int fd, const char* buf, int64_t len, int64_t offset, int64_t start_ts) {
      int err = 0;
      int submit_ret = 0;
      struct iocb io, *p=&io;
      io_prep_pwrite(&io, fd, (void*)buf, len, offset);
      io.data = (void*)start_ts;
      while(true)
      {
        if((submit_ret = io_submit(ctx_, 1, &p)) == 1)
        {
          break;
        }
        else if (-EAGAIN == submit_ret)
        {
          struct io_event e;
          struct timespec timeout;
          timeout.tv_sec=0;
          timeout.tv_nsec=10000;
          if (1 == io_getevents(ctx_, 0, 1, &e, &timeout))
          {
            stat_.add(len, get_usec() - (int64_t)e.data);
          }
        }
        else
        {
          error("io_submit error: %s", strerror(-submit_ret));
          err = submit_ret;
        }
      }
      return err;
    }
  private:
    io_context_t ctx_;
};


int io_stress(IOWorker* worker)
{
  int err = 0;
  int64_t start_time = get_usec();
  int64_t end_time = start_time;
  int64_t stop_time = start_time + cfgi("time_limit", "3000000");
  int64_t write_size = cfgi("write_size", "1024");
  char* write_buf = NULL;
  int64_t pos = 0;
  fprintf(stderr, "io_stress(path=%s, time_limit=%ld, write_size=%ld)\n", cfg("path", "tmp"), stop_time - start_time, write_size);
  if (0 != (err = posix_memalign((void**)&write_buf, 4096, write_size)))
  {
    error("posix_memalign(size=%ld)=>%s", write_size, strerror(errno));
  }
  worker->get_stat().start();
  for(pos = 0; 0 == err && end_time < stop_time; pos += write_size)
  {
    err = worker->submit(pos, write_buf, write_size);
    end_time = worker->get_last_submit_time();
  }
  worker->get_stat().end();
  worker->get_stat().report();
  if (NULL != write_buf)
  {
    free(write_buf);
  }
  return err;
}

int main(int argc, char** argv)
{
  int err = 0;
  const char* path = cfg("path", "tmp");
  int64_t total_size = cfgi("total_size", "1073741824");
  int64_t file_size = cfgi("file_size", "67108864");
  int64_t n_file = total_size/file_size;
  AsyncIOWorker async_worker(path, n_file, file_size);
  SyncIOWorker sync_worker(path, n_file, file_size);
  if (argc != 2)
  {
    fprintf(stderr, _usages, argv[0]);
  }
  else if (0 != (err = io_stress(0 == strcmp(argv[1], "async")? (IOWorker*)&async_worker: (IOWorker*)&sync_worker)))
  {
    error("io_stress()=>%d", err);
  }
  return err;
}
