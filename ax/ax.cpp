#include "nio.h"
#include "utils.h"
#include "pcounter.h"
#include "obj_pool.h"
#include "ax.h"

struct AxPacket: public Packet
{
  AxPacket(): timestamp_(0) {}
  ~AxPacket() {}
  uint64_t timestamp_;
};

class AxHandler: public PacketHandler
{
public:
  typedef ObjPool<AxPacket> PacketPool;
public:
  AxHandler() {}
  virtual ~AxHandler() {}
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
      pkt_pool_.free((AxPacket*)pkt);
    }
    return err;
  }
  int handle_packet(Id id, Packet* req) {
    int err = AX_SUCCESS;
    AxPacket* pkt = (typeof(pkt))req;
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

class AxApp
{
private:
  int fill_open_config(AxStore::OpenConfig& config, const char* workdir, Printer& printer) {
    int err = AX_SUCCESS;
    config.log_path_ = printer.new_str("%s/log.store", workdir);
    config.ballot_path_ = printer.new_str("%s/ballot.store", workdir);
    return err;
  }
  int fill_format_config(AxStore::FormatConfig& config, const char* addr) {
    int err = AX_SUCCESS;
    config.token_.server_.parse(addr);
    config.token_.start_time_ = get_us();
    return err;
  }
public:
  int format(const char* workdir, const char* addr) {
    int err = AX_SUCCESS;
    Printer printer;
    AxStore store;
    AxStore::OpenConfig open_config;
    AxStore::FormatConfig format_config;
    fill_open_config(open_config, workdir, printer);
    fill_format_config(format_config, addr);
    MLOG(INFO, "format: %s %s", workdir, addr);
    if (AX_SUCCESS != (err = store.format(&open_config, &format_config)))
    {
      MLOG(ERROR, "format: %s %s fail: err=%d", workdir, addr, err);
    }
    return err;
  }
  int bootstrap(const char* workdir) {
    int err = AX_SUCCESS;
    MLOG(INFO, "bootstrap: %s", workdir);
    return err;
  }
  int start(const char* workdir) {
    int err = AX_SUCCESS;
    MLOG(INFO, "start: %s", workdir);
    return err;
  }
  int start2(int port)
  {
    int err = AX_SUCCESS;
    int64_t max_pkt_num = 1<<10;
    int64_t max_conn = 1<<16;
    MallocGuard allocator;
    AxHandler handler;
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
        AxPacket pkt;
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
};

const char* __app_name__ = "ax";
const char* __usages__ = "Usages:\n"
  "\t%1$s format workdir addr\n"
  "\t%1$s bootstrap workdir\n"
  "\t%1$s start workdir\n";

#include "cmd_args_parser.h"
#define report_error(err, ...) if (AX_SUCCESS != err)MLOG(ERROR, __VA_ARGS__);
#define define_cmd_call(cmd, ...) if (AX_CMD_ARGS_NOT_MATCH == err && AX_CMD_ARGS_NOT_MATCH != (err = CmdCall(argc, argv, cmd, __VA_ARGS__):AX_CMD_ARGS_NOT_MATCH)) \
  { \
    report_error(err,  "%s() execute fail, ret=%d", #cmd, err);  \
  }

int main(int argc, char** argv)
{
  int err = AX_CMD_ARGS_NOT_MATCH;
  AxApp app;
  define_cmd_call(app.format, StrArg(workdir), StrArg(addr));
  define_cmd_call(app.bootstrap, StrArg(workdir));
  define_cmd_call(app.start, StrArg(workdir));

  if (AX_CMD_ARGS_NOT_MATCH == err)
  {
    fprintf(stderr, __usages__, __app_name__);
  }
  return err;
}
