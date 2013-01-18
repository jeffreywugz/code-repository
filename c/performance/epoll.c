#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <sys/epoll.h>
#include <sys/time.h>
#include <pthread.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#define cfgi(k, v) atoll(getenv(k)?:v)
#define error(...) { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); }
int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

#define DEFAULT_PKT_SIZE "102400"
typedef struct Packet
{
  int64_t send_ts;
  int64_t recv_ts;
  char buf[1024*1000];
} Packet;
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
  printf("rm fd[%d]\n", fd);
  if (0 != epoll_ctl(efd, EPOLL_CTL_DEL, fd, &event))
  {
    err = -errno;
    perror("epoll_ctl(del)");
  }
  return err;
}

ssize_t read_fd(int fd, char* buf, ssize_t total)
{
  ssize_t count = 0;
  ssize_t read_size = 0;
  while(read_size < total && (count = read(fd, buf + read_size, total - read_size)) > 0)
  {
    read_size += count;
  }
  return read_size;
}

ssize_t write_fd(int fd, char* buf, ssize_t total)
{
  ssize_t count = 0;
  ssize_t write_size = 0;
  while(write_size < total && (count = write(fd, buf + write_size, total - write_size)) > 0)
  {
    write_size += count;
  }
  return write_size;
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

#define MAXEPOLLSIZE 10000
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
  //else if ((efd = epoll_create1(0)) <= 0)
  else if ((efd = epoll_create(MAXEPOLLSIZE)) <= 0)
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
  else if (0 != (err = add_fd_to_epoll(efd, sfd, EPOLLIN, 0)))
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

int set_tcp_opt(int fd)
{
  int err = 0;
  int quickAckInt = 1;
  int noDelayInt = 1;
  err = setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK,
                   (const void *)(&quickAckInt), sizeof(quickAckInt));
  err = setsockopt(fd, IPPROTO_TCP, TCP_NODELAY,
                   (const void *)(&noDelayInt), sizeof(noDelayInt));
  return err;
}

int run_server2(int type, in_addr_t ip, int port, void* arg, int64_t timeout_us)
{
  int err = 0;
  int sfd = 0;
  int fd = 0;
  ssize_t count = 0;
  Packet packet;
  ssize_t pkt_size = cfgi("pkt_size", DEFAULT_PKT_SIZE);
  if ((sfd = make_inet_socket(type, ip, port)) <= 0)
  {
    err = -errno;
    error("make_inet_socket()=>%s", strerror(errno));
  }
  else if (0 != set_tcp_opt(sfd))
  {
    error("set_tcp_opt(%d)=>%s", sfd, strerror(errno));
  }
  else if (SOCK_STREAM == type && 0 != (err = listen(sfd, SOMAXCONN)))
  {
    err = -errno;
    error("listen()=>%s", strerror(errno));
  }
    
  while(0 == err)
  {
    if ((fd= accept(sfd, NULL, NULL)) < 0)
    {
      error("accept()=>%s", strerror(errno));
    }
    else
    {
      while(0 == err)
      {
        if (read_fd(fd, (char*)&packet, pkt_size) < pkt_size)
        {
          err = -errno;
          perror ("read_req");
        }
        else
        {
          packet.recv_ts = get_usec();
          if (write_fd(fd, (char*)&packet, pkt_size) < pkt_size)
          {
            err = -errno;
            perror("write_response:");
          }
          printf("receive<%ld:%ld>:%ld,err=%d\n", packet.send_ts, packet.recv_ts, packet.recv_ts - packet.send_ts, err);
        }
      }
      if (0 != close(fd))
      {
        err = -errno;
        perror("close");
      }
      else
      {
        //err = 0;
      }
    }
  }

  if (sfd > 0)
    close(sfd);
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
  Packet packet;
  //fprintf(stderr, "handle event [%d]!\n", events);
  if (events & EPOLLERR)
  {
    fprintf (stderr, "epoll error\n");
    close(fd);
    err = -ENETRESET;
  }
  //else if (events & EPOLLIN || events & EPOLLHUP || events & EPOLLRDHUP)
  else if (events & EPOLLIN || events & EPOLLHUP)
  {
    if (sfd == fd)
    {
      //fprintf(stderr, "handle accept\n");
      if (events & EPOLLHUP)
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
      count = read_fd(fd, (char*)&packet, sizeof(packet));
      if (count < sizeof(packet))
      {
        err = -errno;
        if (EAGAIN != errno && EWOULDBLOCK != errno)
          perror ("read");
      }
      else
      {
        packet.recv_ts = get_usec();
        while(0 > write(fd, (char*)&packet, sizeof(packet)) && errno == EAGAIN)
          ;
        err = (errno == EAGAIN)?0:-errno;
        perror("write_response:");
        printf("receive<%ld:%ld>:%ld,err=%d\n", packet.send_ts, packet.recv_ts, packet.recv_ts - packet.send_ts, err);
      }
      //printf("read(count=%ld)=>%d\n", count, err);
      if (0 == err)
      {}
      else if (0 != (err = rm_fd_from_epoll(efd, fd)))
      {
        error("rm_fd_from_epoll(efd=%d, fd=%d)=>err\n", efd, fd, err);
      }
      else if (0 != close(fd))
      {
        err = -errno;
        perror("close");
      }
    }
  }
  return err;
}

int watch_counter(volatile int64_t* counter, int64_t interval_us, int64_t duration_us, const char* msg)
{
  int64_t start_time_us = get_usec();
  int64_t cur_time_us = 0;
  int64_t old_counter = 0;
  while(1)
  {
    cur_time_us = get_usec();
    if (start_time_us + duration_us < cur_time_us)
      break;
    printf("[%ldus %s]=%ld:%ld\n", cur_time_us - start_time_us, msg, *counter, 1000000* (*counter - old_counter)/interval_us);
    old_counter = *counter;
    usleep(interval_us);
  }
  return 0;
}


struct ClientCfg
{
  in_addr_t ip_;
  int port_;
  volatile int64_t stop_;
  volatile int64_t npkts_send_;
  volatile int64_t total_time_;
};
struct ClientCfg client_cfg;

int run_client(int type, in_addr_t ip, int port, int64_t id, volatile int64_t* stop, volatile int64_t* npkts_send)
{
  int err = 0;
  char id_buf[16];
  int fd = 0;
  Packet packet;
  ssize_t pkt_size = cfgi("pkt_size", DEFAULT_PKT_SIZE);
  int64_t cur_ts = 0;
  int64_t send_ts = 0;
  int64_t client_wait_us = cfgi("client_wait_us", "100000");
  struct sockaddr_in sin;
  sin.sin_port = htons(port);
  sin.sin_addr.s_addr = ip;
  sin.sin_family = AF_INET;

  if (type != SOCK_STREAM || NULL == stop)
  {
    err = -EINVAL;
  }
  else if (0 >= snprintf(id_buf, sizeof(id_buf), "%ld", id))
  {
    err = -1;
  }
  else if ((fd = socket(AF_INET, type, 0)) < 0)
  {
    perror("socket");
    err = -errno;
  }
  else if (0 != set_tcp_opt(fd))
  {
    error("set_tcp_opt(%d)=>%s", fd, strerror(errno));
  }
  else if (0 != (connect(fd, (struct sockaddr *)&sin, sizeof(sin))))
  {
    perror("connect");
    err = -errno;
  }
  for(int64_t i = 0; !*stop && 0 == err; i++)
  {
    send_ts = get_usec();
    packet.send_ts = send_ts;
    if (write_fd(fd, (char*)&packet, pkt_size) < pkt_size)
    {
      perror("write_req");
      err = -errno;
    }
    else if (read_fd(fd, (char*)&packet, pkt_size) < pkt_size)
    {
      perror("read_response");
      break;
    }
    else
    {
      cur_ts = get_usec();
      if (send_ts != packet.send_ts)
      {
        printf("packet.send_ts[%ld] != send_ts[%ld]\n", packet.send_ts, send_ts);
      }
      client_cfg.total_time_ += cur_ts - packet.send_ts;
      printf("client<%ld:%ld:%ld>:%ld:%ld\n", packet.send_ts, packet.recv_ts, cur_ts,
             packet.recv_ts - packet.send_ts, cur_ts - packet.send_ts);
    }
    __sync_fetch_and_add(npkts_send, 1);
    usleep(client_wait_us);
    //printf("client(i=%ld, id=%ld, packets=%ld)=>%d\n", i, id, n_packets, err);
  }
  if (fd > 0 && 0 != (close(fd)))
  {
    perror("close");
    err = -errno;
  }
  return err;
}

int test_server(in_addr_t ip, int port)
{
  int64_t timeout_us = 10*1000;
  printf("server(port=%d)\n", port);
  return run_server2(SOCK_STREAM, ip, port, NULL, timeout_us);
}

int launch_client(struct ClientCfg* cfg)
{
  return run_client(SOCK_STREAM, cfg->ip_, cfg->port_, 0, &cfg->stop_, &cfg->npkts_send_);
}

int test_client(in_addr_t ip, int port, const int64_t n_conn)
{
  int err = 0;
  int64_t duration = cfgi("duration", "1000000");
  pthread_t threads[n_conn];
  struct in_addr addr;
  addr.s_addr = ip;
  client_cfg.ip_ = ip;
  client_cfg.port_ = port;
  client_cfg.npkts_send_ = 0;
  client_cfg.total_time_ = 0;
  printf("client(addr=%s:%d, n_conn=%ld, duration=%ld)\n", inet_ntoa(addr), port, n_conn, duration);
  for(int64_t i = 0; 0 == err && i < n_conn; i++)
  {
    if (0 != (pthread_create(threads + i, NULL, launch_client, &client_cfg)))
    {
      err = -EINVAL;
      error("launch_client(addr=%s:%d, id=%ld)=>%d", inet_ntoa(addr), port, i, err);
    }
  }
  if (0 == err)
  {
    watch_counter(&client_cfg.npkts_send_, 1000000, duration, "pkts");
    client_cfg.stop_ = 1;
    printf("avg_latency: %ld/%ld=%ld\n", client_cfg.total_time_, client_cfg.npkts_send_,
           client_cfg.npkts_send_? client_cfg.total_time_/client_cfg.npkts_send_:0);
  }
  for(int64_t i = 0; 0 == err && i < n_conn; i++)
  {
    if (0 != (pthread_join(threads[i], NULL)))
    {
      err = -EINVAL;
      error("wait_client(addr=%s:%d, id=%ld)=>%d", inet_ntoa(addr), port, i, err);
    }
  }
  return err;
}

int main(int argc, char **argv)
{
  int err = 0;
  bool show_help = false;
  if (argc < 2)
  {
    err = -EINVAL;
    show_help = true;
  }
  else if (0 == strcmp(argv[1], "server"))
  {
    if (argc != 4)
    {
      err = -EINVAL;
      show_help = true;
    }
    else if (0 != (err = test_server(inet_addr(argv[2]), atoi(argv[3]))))
    {
      error("test_server(ip=%s, port=%s)=>%d", argv[2], argv[3], err);
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
    fprintf(stderr, "Usage:\n\t%1$s server ip port\n" "\t%1$s client ip port n_conn\n", argv[0]);
  }

  return 0;
}
