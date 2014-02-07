#include "ax.h"

class AxApp
{
public:
  int bootstrap(const char* workdir);
  int start(const char* workdir);
  int start_group(const char* leader, const char* server_list);
  int stop_group(const char* leader, const char* server_list);
  int propose(const char* leader, const char* cursor, const char* content);
  int read(const char* server, const char* cursor);
protected:
  AxServer* get_server();
  AxClient* get_client();
};

const char* __usage__ = "Usages:"
  "ax start workdir"
  "ax bootstrap workdir"
  "ax start_group leader_ip:leader_port ip1:port1,ip2:port2..."
  "ax stop_group leader_ip:leader_port ip1:port1,ip2:port2..."
  "ax propose leader_ip:leader_port term:pos log_content"
  "ax read ip:port term:pos -> ";

#define report_error(err, ...) if (AX_SUCCESS != err)AX_LOG(ERROR, __VA_ARGS__);
#include "cmd_args_parser.h"

int main(int argc, char** argv)
{
  int err = 0;
  AxApp app;
  if (AX_CMD_ARGS_NOT_MATCH != CmdCall(argc, argv, app.bootstrap, StrArg(workdir)):AX_CMD_ARGS_NOT_MATCH)
  {
    report_error(err, "malloc_stress()=>%d", err);
  }

  return err;
}
