struct fepoll_t
{
};
static int make_fd_non_blocking(int fd)
{
  int err = 0;
  int flags = 0;
  if ((flags = fcntl(fd, F_GETFL, 0)) < 0 || fcntl(fd, F_SETFL, flags|O_NONBLOCK) < 0)
  {
    perror ("fcntl:");
    err = -errno;
  }
  return err;
}

static int make_inet_socket(int type, in_addr_t ip, int port)
{
  int err = 0;
  int fd = -1;
  struct sockaddr_in sin;
  sin.sin_port = htons(port);
  sin.sin_addr.s_addr = ip;
  sin.sin_family = AF_INET;
  if ((fd = socket(AF_INET, type, 0)) < 0
      || bind(fd, (struct sockaddr *)&sin,sizeof(struct sockaddr_in)) != 0)
  {
    perror("socket");
    err = -errno;
  }
  if (fd > 0 && 0 != err)
  {
    close(fd);
  }
  return 0 == err? fd: -1;
}

static int add_fd_to_epoll(int efd, int fd, uint32_t events, uint32_t session_id)
{
  int err = 0;
  struct epoll_event event;
  uint64_t data = fd;
  data = (data<<32) | session_id;
  event.data.u64 = data;
  event.events = events;
  if (0 != (err = make_fd_non_blocking(fd)))
  {
    error("make_fd_non_blocking()=>%s", strerror(err));
  }
  else if (0 != epoll_ctl(efd, EPOLL_CTL_ADD, fd, &event))
  {
    err = -errno;
    error("epoll_ctl()=>%s", strerror(err));
  }
  return err;
}

static int rm_fd_from_epoll(int efd, int fd)
{
  int err = 0;
  struct epoll_event event;
  printf("rm fd[%d]\n", fd);
  if (0 != epoll_ctl(efd, EPOLL_CTL_DEL, fd, &event))
  {
    err = -errno;
    perror("epoll_ctl(del)");
  }
  return err;
}

fepoll_t* fepoll_create(int64_t size, int64_t nthread);
void fepoll_destroy(fepoll_t* fepoll);
void fepoll_add(fepoll_t* fepoll, int fd, struct epoll_event* event);
struct epoll_event* fepoll_get(fepoll_t* fepoll);
