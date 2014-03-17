#include "nio.h"
#include "utils.h"
#include "pcounter.h"
#include "obj_pool.h"

class EchoHandler: public PacketHandler
{
public:
  typedef ObjPool<Packet> PacketPool;
public:
  EchoHandler() {}
  virtual ~EchoHandler() {}
public:
  int init(Nio* nio, int64_t max_pkt_num) {
    int err = AX_SUCCESS;
    nio_ = nio;
    err = pkt_pool_.init(max_pkt_num, allocator_.alloc(pkt_pool_.calc_mem_usage(max_pkt_num)));
    return err;
  }
  int alloc_packet(Packet*& pkt) {
    int err = AX_SUCCESS;
    if (NULL == (pkt = pkt_pool_.alloc()))
    {
      err = AX_NO_MEM;
    }
    return err;
  }
  int free_packet(Packet* pkt) {
    int err = AX_SUCCESS;
    if (NULL != pkt)
    {
      pkt_pool_.free(pkt);
    }
    return err;
  }
  int handle_packet(Id id, Packet* pkt) {
    int err = AX_SUCCESS;
    if (INVALID_ID == id || NULL == pkt)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL == nio_)
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS != (err = nio_->send_packet(id, pkt)))
    {
      MLOG(WARN, "send_packet()=>%d", err);
    }
    if (NULL != pkt)
    {
      PC_ADD(RPKT, 1);
      //MLOG(INFO, "receive pkt: len=%d content=%s", pkt->len_, pkt->payload_);
    }
    return err;
  }
private:
  Nio* nio_;
  MallocGuard allocator_;
  PacketPool pkt_pool_;
};

class EchoServer
{
public:
  EchoServer(): stop_(false), thread_num_(0) {}
  ~EchoServer(){}
public:
  int init(int port, int thread_num) {
    int err = AX_SUCCESS;
    int64_t max_pkt_num = 1<<10;
    int64_t capacity = 1<<16;
    if (AX_SUCCESS != (err = handler_.init(&nio_, max_pkt_num)))
    {
      MLOG(ERROR, "handler.init()=>%d", err);
    }
    else if (AX_SUCCESS != (err = nio_.init(&handler_, port, capacity, (char*)ax_alloc(nio_.calc_mem_usage(capacity)))))
    {
      MLOG(ERROR, "io_sched.init()=>%d", err);
    }
    else
    {
      thread_num_ = thread_num;
      MLOG(INFO, "echo_server init: port=%d thread_num=%d", port, thread_num);
    }
    return err;
  }
  int mainloop() {
    int err = AX_SUCCESS;
    while(!stop_) {
      nio_.sched(100 * 1000);
      PC_REPORT();
    }
    return err;
  }
  int start() {
    int sys_err = 0;
    int err = AX_SUCCESS;
    for(int64_t i = 0; i < min(thread_num_, (int64_t)arrlen(thread_)); i++) {
      if (0 != (sys_err = pthread_create(thread_ + i, NULL, (void* (*)(void*))thread_func, (void*)this)))
      {
        err = AX_FATAL_ERR;
        MLOG(ERROR, "pthread_create fail, err=%d", sys_err);
      }
    }
    return err;
  }
  int stop() {
    stop_ = true;
    return AX_SUCCESS;
  }
  int wait() {
    for(int64_t i = 0; i < min(thread_num_, (int64_t)arrlen(thread_)); i++) {
      pthread_join(thread_[i], NULL);
    }
    return AX_SUCCESS;
  }
  static int thread_func(EchoServer* self) {
    return self->mainloop();
  }
  Nio& get_nio() { return nio_; }
private:
  bool stop_;
  int64_t thread_num_;
  pthread_t thread_[32];
  Nio nio_;
  EchoHandler handler_;
};

int test_echo_server(int port, int thread_num)
{
  int err = AX_SUCCESS;
  EchoServer server;
  if (AX_SUCCESS != (err = server.init(port, thread_num)))
  {
    MLOG(ERROR, "echo server init fail, err=%d", err);
  }
  else if (AX_SUCCESS != (err = server.start()))
  {
    MLOG(ERROR, "start()=>%d", err);
  }
  else
  {
    Server server_addr;
    server_addr.ip_ = inet_addr("127.0.0.1");
    server_addr.port_ = port;
    char buf[32] = "hello,world!";
    Packet pkt;
    pkt.set_buf(buf, strlen(buf));
    if (AX_SUCCESS != (err = server.get_nio().send_packet(server_addr, &pkt)))
    {
      MLOG(ERROR, "send_first packet");
    }
    else
    {
      MLOG(INFO, "send first packet");
    }
    server.wait();
  }
  return err;
}

int main(int argc, char** argv)
{
  int err = AX_CMD_ARGS_NOT_MATCH;
  const char* usages = "Usages:\n ./echo_server port n_threads\n";
  int port = 0;
  int n_threads = 1;
  if (argc != 3)
  {}
  else
  {
    port = atoi(argv[1]);
    n_threads = atoi(argv[2]);
  }
  if (port <= 0 || n_threads <= 0)
  {
    fprintf(stderr, usages);
  }
  else
  {
    err = test_echo_server(port, n_threads);
  }
  return err;
}
