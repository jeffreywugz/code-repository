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

int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

struct IOStat
{
  IOStat(): total_write_bytes_(0), total_write_count_(0), total_write_time_(0) {}
  ~IOStat(){}
  void add(int64_t write_size, int64_t write_time){
    __sync_fetch_and_add(&total_write_count_, 1);
    __sync_fetch_and_add(&total_write_bytes_, write_size);
    __sync_fetch_and_add(&total_write_time_, write_time);
  }
  void report() {
    fprintf(stderr, "bw=%ldB/%ldus=%lfM/s, iops=%ld/%ld=%ldIO/s\n",
            total_write_bytes_, total_write_time_, 1.0 * total_write_bytes_/total_write_time_,
            total_write_count_, total_write_time_, 1000000 * total_write_count_/total_write_time_);
  }
  int64_t total_write_bytes_;
  int64_t total_write_count_;
  int64_t total_write_time_;
};

class IOWorker
{
  public:
    IOWorker(const char* path, int64_t n_file, int64_t file_size): path_(path), n_file_(n_file), file_size_(file_size),
                                                                   fd_(-1), last_file_no_(-1){}
    virtual ~IOWorker() {}
    int submit(int64_t pos, char* buf, int64_t len) {
      int err = 0;
      int64_t file_no = (pos/file_size_) % n_file_;
      int fd = -1;
      if (pos < 0 || NULL == buf || len < 0)
      {
        err = -EINVAL;
      }
      else if (0 > (fd = get_fd(pos)))
      {
        error("get_fd(%s) faile", pos);
      }
      else if (0 != do_io(fd, buf, len, pos % file_size_))
      {
        error("do_io(pos=%ld) fail", pos);
      }
      return err;
    }

    virtual int do_io(int fd, const char* buf, int64_t len, int64_t offset) {
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
        }
      }
      return fd_;
    }
  private:
    const char* path_;
    int64_t n_file_;
    int64_t file_size_;
    int64_t fd_;
    int64_t last_file_no_;
};

class SyncIOWorker: public IOWorker
{
  public:
    SyncIOWorker(const char* path, int64_t n_file, int64_t file_size): IOWorker(path, n_file, file_size) {}
    virtual ~SyncIOWorker() {}
    int do_io(int fd, const char* buf, int64_t len, int64_t offset) {
      int err = 0;
       if (len != pwrite(fd, buf, len, offset))
       {
         err = -EIO;
       }
       else if (0 != fsync(fd))
       {
         err = -EIO;
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
    
    int do_io(int fd, const char* buf, int64_t len, int64_t offset) {
      int err = 0;
      int submit_ret = 0;
      struct iocb io, *p=&io;
      io_prep_pwrite(&io, fd, (void*)buf, len, offset);
      io.data = (void*)buf;
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
          io_getevents(ctx_, 0, 1, &e, &timeout);
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

int io_stress(IOWorker* worker, IOStat* stat)
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
  for(pos = 0; 0 == err && end_time < stop_time; pos += write_size)
  {
    start_time = end_time;
    err = worker->submit(pos, write_buf, write_size);
    end_time = get_usec();
    stat->add(write_size, end_time - start_time);
  }
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
  int64_t n_file = cfgi("n_file", "20");
  int64_t file_size = cfgi("file_size", "67108864");
  AsyncIOWorker async_worker(path, n_file, file_size);
  SyncIOWorker sync_worker(path, n_file, file_size);
  IOStat stat;
  if (argc != 2)
  {
    fprintf(stderr, _usages, argv[0]);
  }
  else if (0 != (err = io_stress(0 == strcmp(argv[1], "async")? (IOWorker*)&async_worker: (IOWorker*)&sync_worker, &stat)))
  {
    error("io_stress()=>%d", err);
  }
  else
  {
    stat.report();
  }
  return err;
}
