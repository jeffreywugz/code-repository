class Epoll
{
  struct Event
  {
    int fd_;
    epoll_event event_;
  };
public:
  Epoll(): epfd_(-1) {}
  ~Epoll(){}
public:
  int init(int64_t size, int64_t nthread){
    int err = 0;
    if ((epfd_ = epoll_create1(EPOLL_CLOEXEC)) < 0)
    {
      err = errno;
    }
    return err;
  }
  void destroy() {
    if (epfd_ >= 0)
    {
      close(epfd_);
    }
  }
  int add(int fd, struct epoll_event* event);
  struct epoll_event* get() {
    int epoll_wait(int epfd, struct epoll_event *events,
                   int maxevents, int timeout);
  }
private:
  int epfd_;
  Fixed
};
