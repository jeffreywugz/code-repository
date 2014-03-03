
class EvHandler
{
public:
  typedef struct epoll_event Event;
  typedef int (*ev_callback_t)(void* arg1, void* arg2, Event* event);
  struct Callback
  {
    Callback(): callback_(NULL), arg1_(NULL), arg2_(NULL) {}
    ~Callback() {}
    int do_callback(Event* event) { return NULL != callback_? callback_(arg1_, arg2_, fd_, event): AX_CALLBACK_NOT_SET; }
    int fd_;
    Event event_;
    ev_callback_t callback_;
    void* arg1_;
    void* arg2_;
  };
public:
  EvHandler(): epoll_(NULL) {}
  ~EvHandler() {}
public:
  int init(Epoll* epoll, int64_t capacity) {
    int err = AX_SUCCESS;
    if (NULL == epoll || capacity <= 0)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL != epoll_)
    {
      err = AX_INIT_TWICE;
    }
    else if (AX_SUCCESS != (err = cbmgr_.init(capacity)))
    {}
    return err;
  }
  int add_callback(Callback& callback) {
    int err = AX_SUCCESS;
    if (AX_SUCCESS != (err = cbmgr_.add(id, callback)))
    {}
    else
    {
      callback.events_.data.u64 = id;
      err = epoll_->add(callback.fd_, callback.event_);
    }
    return err;
  }
  int handle_event(Event* event) {
    int err = AX_SUCCESS;
    int cberr = AX_SUCCESS;
    uint64_t id = 0;
    Callback callback;
    if (NULL == epoll_)
    {
      err = AX_NOT_INIT;
    }
    else if (NULL == event)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (AX_SUCCESS != (err = cbmgr_.get((id = event->data.u64), callback)))
    {}
    else if (AX_SUCCESS != (cberr = callback.do_callback(event)))
    {
      cbmgr_.rm(id, callback);
      close(callback.fd_);
    }
    else
    {
      callback.events_.data.u64 = id;
      epoll_->add(callback->fd_, callback.event_);
    }
    return err;
  }
private:
  Epoll* epoll_;
  IDMap<Callback> cbmgr_;
};
