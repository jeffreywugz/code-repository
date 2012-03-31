#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/epoll.h>
#include <sys/time.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define error(...) { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); }
int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

#define profile(expr) ({                        \
  int64_t old_us = 0;                                  \
  int64_t new_us = 0;                                  \
  int64_t result = 0;                                    \
  old_us = get_usec();                                   \
  result = expr;                                         \
  new_us = get_usec();                                                  \
  printf("%s=>%ld in %ldms\n", #expr, result, new_us - old_us);           \
  new_us - old_us; })

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
  if (0 != epoll_ctl(efd, EPOLL_CTL_DEL, fd, &event))
  {
    err = -errno;
    perror("epoll_ctl(del)");
  }
  return err;
}

typedef int (*epoll_event_handler_t)(void* self, int efd, int sfd, int fd, uint32_t session_id, uint32_t events);

#define high_32bits(x) (x>>32) & 0xffffffff
#define low_32bits(x) (x>>32) & 0xffffffff
int handle_epool_event(int efd, int sfd, epoll_event_handler_t handler, void* self, struct epoll_event* events, int64_t max_events, int64_t timeout_us)
{
  int err = 0;
  int n = 0, i = 0;
  while(0 == err || -EAGAIN == err)
  {
    n = epoll_wait(efd, events, max_events, timeout_us/1000);
    err = handler(self, efd, sfd, 0, 0, 0);
    for (i = 0; 0 == err && i < n; i++)
    {
      err = handler(self, efd, sfd, high_32bits(events[i].data.u64), low_32bits(events[i].data.u64), events[i].events);
      if (-EAGAIN == err || -ENETRESET == err)
      {
        err = 0;
      }
    }
  }
  return err;
}

#define MAXEVENTS 64
int run_server(int type, in_addr_t ip, int port, epoll_event_handler_t handler, void* arg, int64_t timeout_us)
{
  int err = 0;
  int efd = 0;
  int sfd = 0;
  struct epoll_event* events = NULL;
  if (NULL == (events = calloc(MAXEVENTS, sizeof(*events))))
  {
    error("calloc()=>NULL");
  }
  else if ((efd = epoll_create1(0)) <= 0)
  {
    err = -errno;
    error("epoll_create1()=>%s", strerror(err));
  }
  else if ((sfd = make_inet_socket(type, ip, port)) <= 0)
  {
    err = -errno;
    error("make_inet_socket()=>%s", strerror(errno));
  }
  else if (SOCK_STREAM == type && 0 != (err = listen(sfd, SOMAXCONN)))
  {
    err = -errno;
    error("listen()=>%s", strerror(errno));
  }
  else if (0 != (err = add_fd_to_epoll(efd, sfd, EPOLLIN | EPOLLET, 0)))
  {
    error("add_fd_to_epoll()=>%s", strerror(errno));
  }
  else if (0 != (err = handle_epool_event(efd, sfd, handler, arg, events, MAXEVENTS, timeout_us)))
  {
    error("handle_socket_event()=>%d", err);
  }

  if (NULL != events)
    free(events);
  if (sfd > 0)
    close(sfd);
  if (efd > 0)
    close(efd);
  return err;
}

int handle_tcp_socket_event(void* self, int efd, int sfd, int fd, uint32_t session_id, uint32_t events)
{
  int err = 0;
  struct sockaddr in_addr;
  socklen_t in_len = 0;
  int infd = 0;
  ssize_t count = 0;
  char buf[512];
  //fprintf(stderr, "handle event [%d]!\n", events);
  if (events & EPOLLERR)
  {
    fprintf (stderr, "epoll error\n");
    close(fd);
    err = -ENETRESET;
  }
  else if (events & EPOLLIN || events & EPOLLHUP || events & EPOLLRDHUP)
  {
    if (sfd == fd)
    {
      //fprintf(stderr, "handle accept\n");
      if (events & EPOLLHUP || events & EPOLLRDHUP)
      {
        err = -ENETRESET;
      }
      while (0 == err)
      {
        in_len = sizeof(in_addr);
        if (0 > (infd = accept(sfd, &in_addr, &in_len)))
        {
          err = -errno;
          if (EAGAIN != errno && EWOULDBLOCK != errno)
            perror ("accept");
        }
        else if (0 != (err = add_fd_to_epoll(efd, infd, EPOLLIN|EPOLLET, session_id)))
        {
        }
      }
    }
    else
    {
      //fprintf(stderr, "handle read\n");
      while (0 == err)
      {
        count = read(fd, buf, sizeof(buf));
        if (count == -1)
        {
          err = -errno;
          if (EAGAIN != errno && EWOULDBLOCK != errno)
            perror ("read");
        }
        else if (count == 0)
        {
          err = -EAGAIN;
        }
        else
        {
          printf("%s\n", buf);
        }
      }
      if (0 >= count && -EAGAIN != err &&  0 != (err = rm_fd_from_epoll(efd, fd)))
      {
        error("rm_fd_from_epoll(efd=%d, fd=%d)=>err\n", efd, fd, err);
      }
    }
  }
  return err;
}

int run_client(int type, in_addr_t ip, int port, int64_t id)
{
  int err = 0;
  char id_buf[16];
  int fd = 0;
  struct sockaddr_in sin;
  sin.sin_port = htons(port);
  sin.sin_addr.s_addr = ip;
  sin.sin_family = AF_INET;

  if (type != SOCK_STREAM)
  {
    err = -EINVAL;
  }
  else if (0 >= snprintf(id_buf, sizeof(id_buf), "%llx", id))
  {
    err = -1;
  }
  else if ((fd = socket(AF_INET, type, 0)) < 0)
  {
    perror("socket");
    err = -errno;
  }
  else if (0 != (connect(fd, (struct sockaddr *)&sin, sizeof(sin))))
  {
    perror("connect");
    err = -errno;
  }
  else if (0 >= (write(fd, id_buf, strlen(id_buf) + 1)))
  {
    perror("write");
    err = -errno;
  }
  if (fd > 0 && 0 != (close(fd)))
  {
    perror("close");
    err = -errno;
  }
  return err;
}

int test_server(int port)
{
  int64_t timeout_us = 10*1000;
  printf("server(port=%d)\n", port);
  return run_server(SOCK_STREAM, inet_addr("127.0.0.1"), port, handle_tcp_socket_event, NULL, timeout_us);
}

int test_client(in_addr_t ip, int port, int64_t n_iters)
{
  int err = 0;
  struct in_addr addr;
  addr.s_addr = ip;
  printf("client(addr=%s:%d, n_iters=%ld)\n", inet_ntoa(addr), port, n_iters);
  for(int64_t i = 0; 0 == err && i < n_iters; i++)
  {
    if (0 != (err = run_client(SOCK_STREAM, ip, port, i)))
    {
      error("run_client(addr=%s:%d, id=%ld)=>%d", inet_ntoa(addr), port, i, err);
    }
  }
  return err;
}

int main(int argc, char **argv)
{
  int err = 0;
  bool show_help = false;
  int port = 0;
  int64_t timeout_us = 1000000;
  int64_t n = 0;
  if (argc < 2)
  {
    err = -EINVAL;
    show_help = true;
  }
  else if (0 == strcmp(argv[1], "server"))
  {
    if (argc != 3)
    {
      err = -EINVAL;
      show_help = true;
    }
    else if (0 != (err = test_server(atoi(argv[2]))))
    {
      error("test_server(port=%s)=>%d", argv[2], err);
    }
  }
  else if (0 == strcmp(argv[1], "client"))
  {
    if (argc != 5)
    {
      err = -EINVAL;
      show_help = true;
    }
    else if (0 != (err = test_client(inet_addr(argv[2]), atoi(argv[3]), atoll(argv[4]))))
    {
      error("test_client(%s:%s)=>%d", argv[2], argv[3]);
    }
  }
  else
  {
    err = -EINVAL;
    show_help = true;
  }
  if (show_help)
  {
    fprintf(stderr, "Usage:\n\t%1$s server port\n" "\t%1$s client ip port n_iters\n", argv[0]);
  }

  port = atoi(argv[1]);
  n = 1<<atoi(argv[2]);
  if (port <= 0 || n <= 0)
  {
    return -1;
  }
  return 0;
}
