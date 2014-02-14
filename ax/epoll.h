class Epoll
{
public:
  typedef struct epoll_event Event;
  typedef ObjQueue<Event> EventQueue;
  enum {MAX_EVENT_COUNT = 32};
public:
  Epoll(): epfd_(-1) {}
  ~Epoll(){}
public:
  int init(int64_t qsize){
    int err = AX_SUCCESS;
    Event event;
    event.u64 = ~(0UL);
    if (qsize <= MAX_EVENT_COUNT)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if ((epfd_ = epoll_create1(EPOLL_CLOEXEC)) < 0)
    {
      err = AX_EPOLL_CREATE_ERR;
    }
    else if (AX_SUCCESS != (err = queue_.init(qsize)))
    {}
    else if (AX_SUCCESS != (err = queue_.push(&event)))
    {}
    return err;
  }
  void destroy() {
    if (epfd_ >= 0)
    {
      close(epfd_);
    }
  }

  int add(int fd, Event* event) {
    int err = AX_SUCCESS;
    if (0 != epoll_ctl(epfd_, EPOLL_CTL_ADD, fd, event))
    {
      if (EEXIST != errno)
      {
        err = AX_EPOLL_CTL_ERR;
      }
    }
    return err;
  }
  
  int get(Event* event, int64_t timeout_us) {
    int err = AX_SUCCESS;
    if (NULL == event)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (AX_SUCCESS != (err = queue_.pop(event, timeout_us)))
    {}
    else if (~(0UL) == event->data.u64)
    {
      Event events[MAX_EVENT_COUNT];
      int ready_cnt = 0;
      if ((ready_cnt = epoll_wait(epfd_, events, array_len(events), timeout_us)) < 0)
      {
        err = AX_EPOLL_WAIT_ERR;
      }
      for(int i = 0; AX_SUCCESS == err && i < ready_cnt; i++)
      {
        err = queue_.push(events + i);
      }
    }
    return err;
  }
private:
  int epfd_;
  EventQueue queue_;
};
