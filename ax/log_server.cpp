#include "nio.h"
#include "utils.h"
#include "pcounter.h"
#include "obj_pool.h"

struct LogPacket: public Packet
{
  LogPacket(): timestamp_(0) {}
  ~LogPacket() {}
  uint64_t timestamp_;
};

class LogHandler: public PacketHandler
{
public:
  typedef ObjPool<LogPacket> PacketPool;
public:
  LogHandler() {}
  virtual ~LogHandler() {}
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
      pkt_pool_.free((LogPacket*)pkt);
    }
    return err;
  }
  int handle_packet(Id id, Packet* req) {
    int err = AX_SUCCESS;
    LogPacket* pkt = (typeof(pkt))req;
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

int test_log_server(int port)
{
  int err = AX_SUCCESS;
  int64_t max_pkt_num = 1<<10;
  int64_t max_conn = 1<<16;
  MallocGuard allocator;
  LogHandler handler;
  Nio nio;
  NioDriver nio_driver;
  ThreadWorker thread_worker;
  if (AX_SUCCESS != (err = handler.init(&nio, max_pkt_num)))
  {
    MLOG(ERROR, "handler.init()=>%d", err);
  }
  else if (AX_SUCCESS != (err = nio.init(&handler, port, max_conn, allocator.alloc(nio.calc_mem_usage(max_conn)))))
  {
    MLOG(ERROR, "nio.init()=>%d", err);
  }
  else if (AX_SUCCESS != (err = nio_driver.init(&nio)))
  {
    MLOG(ERROR, "log server init fail, err=%d", err);
  }
  else if (AX_SUCCESS != (err = thread_worker.start(&nio_driver, 1)))
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
      LogPacket pkt;
      pkt.pcode_ = AX_CLOCK_GET;
      pkt.timestamp_ = 0;
      if (AX_SUCCESS != (err = nio.send_packet(server_addr, &pkt)))
      {
        MLOG(ERROR, "get_log request: err=%d", err);
      }
      usleep(1000);
    }
  }
  return err;
}

int main(int argc, char** argv)
{
  int err = AX_CMD_ARGS_NOT_MATCH;
  const char* usages = "Usages:\n ./log_server port\n";
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
    err = test_log_server(port);
  }
  return err;
}
