#include "nio.h"
#include "utils.h"
#include "pcounter.h"
#include "obj_pool.h"

struct ClockPacket: public Packet
{
  ClockPacket(): timestamp_(0) {}
  ~ClockPacket() {}
  uint64_t timestamp_;
};

class ClockHandler: public PacketHandler
{
public:
  typedef ObjPool<ClockPacket> PacketPool;
public:
  ClockHandler() {}
  virtual ~ClockHandler() {}
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
      pkt_pool_.free((ClockPacket*)pkt);
    }
    return err;
  }
  int handle_packet(Id id, Packet* req) {
    int err = AX_SUCCESS;
    ClockPacket* pkt = (typeof(pkt))req;
    if (INVALID_ID == id || NULL == pkt)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL == nio_)
    {
      err = AX_NOT_INIT;
    }
    else if (AX_CLOCK_GET == pkt->pcode_)
    {
      pkt->pcode_ = AX_RESPONSE;
      pkt->timestamp_ = get_us();
      if (AX_SUCCESS != (err = nio_->send_packet(id, pkt)))
      {
        MLOG(WARN, "send_packet()=>%d", err);
      }
      PC_ADD(RPKT, 1);
    }
    return err;
  }
private:
  Nio* nio_;
  MallocGuard allocator_;
  PacketPool pkt_pool_;
};

class ClockServer: public IThreadCallable
{
public:
  ClockServer(): stop_(false) {}
  ~ClockServer(){}
public:
  int init(int port) {
    int err = AX_SUCCESS;
    int64_t max_pkt_num = 1<<3;
    int64_t capacity = 1<<16;
    if (AX_SUCCESS != (err = handler_.init(&nio_, max_pkt_num)))
    {
      MLOG(ERROR, "handler.init()=>%d", err);
    }
    else if (AX_SUCCESS != (err = nio_.init(&handler_, port, capacity, allocator_.alloc(nio_.calc_mem_usage(capacity)))))
    {
      MLOG(ERROR, "io_sched.init()=>%d", err);
    }
    else
    {
      MLOG(INFO, "clock_server init: port=%d thread_num=%d", port);
    }
    return err;
  }
  int do_thread_work() {
    int err = AX_SUCCESS;
    while(!stop_) {
      nio_.sched(100 * 1000);
      PC_REPORT();
    }
    return err;
  }
  Nio& get_nio() { return nio_; }
private:
  bool stop_;
  MallocGuard allocator_;
  Nio nio_;
  ClockHandler handler_;
};

int test_clock_server(int port)
{
  int err = AX_SUCCESS;
  ClockServer clock_server;
  ThreadWorker thread_worker;
  if (AX_SUCCESS != (err = clock_server.init(port)))
  {
    MLOG(ERROR, "clock server init fail, err=%d", err);
  }
  else if (AX_SUCCESS != (err = thread_worker.start(&clock_server, 1)))
  {
    MLOG(ERROR, "start server, err=%d", err);
  }
  else
  {
    Server server_addr;
    server_addr.ip_ = inet_addr("127.0.0.1");
    server_addr.port_ = port;
    for(int64_t i = 0; i < 1024000; i++)
    {
      ClockPacket pkt;
      pkt.pcode_ = AX_CLOCK_GET;
      pkt.timestamp_ = 0;
      if (AX_SUCCESS != (err = clock_server.get_nio().send_packet(server_addr, &pkt)))
      {
        MLOG(ERROR, "get_clock request: err=%d", err);
      }
      usleep(1000);
    }
  }
  return err;
}

int main(int argc, char** argv)
{
  int err = AX_CMD_ARGS_NOT_MATCH;
  const char* usages = "Usages:\n ./clock_server port\n";
  int port = 0;
  if (argc != 2)
  {}
  else
  {
    port = atoi(argv[1]);
  }
  if (port <= 0)
  {
    fprintf(stderr, usages);
  }
  else
  {
    err = test_clock_server(port);
  }
  return err;
}
