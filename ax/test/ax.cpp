#include "ax.h"

class AxApp
{
public:
  int bootstrap(const char* workdir) {
    int err = AX_SUCCESS;
    AxLogServer* server = NULL;
    if (NULL == (server = get_server()))
    {
      err = AX_NO_MEM;
    }
    else if (AX_SUCCESS != (err = server->init(workdir)))
    {
      MLOG(ERROR, "server init fail, err=%d", err);
    }
    else if (AX_SUCCESS != (err = server->bootstrap()))
    {
      MLOG(ERROR, "server bootstrap fail, err=%d", err);
    }
    return err;
  }
  int start(const char* workdir) {
    int err = AX_SUCCESS;
    AxLogServer* server = NULL;
    if (NULL == (server = get_server()))
    {
      err = AX_NO_MEM;
    }
    else if (AX_SUCCESS != (err = server->init(workdir)))
    {
      MLOG(ERROR, "server init fail, err=%d", err);
    }
    else if (AX_SUCCESS != (err = server->start()))
    {
      MLOG(ERROR, "server start fail, err=%d", err);
    }
    return err;
  }

  int start_group(const char* leader_spec, const char* server_list_spec) {
    int err = AX_SUCCESS;
    AxAdminClient* client = NULL;
    Server leader;
    ServerList server_list;
    if (AX_SUCCESS != (err = parse(leader, leader_spec)))
    {
      MLOG(ERROR, "invalid leader: %s", leader_spec);
    }
    else if (AX_SUCCESS != (err = parse(server_list, server_list_spec)))
    {
      MLOG(ERROR, "invalid server_list: %s", server_list_spec);
    }
    else if (NULL == (client = get_client()))
    {
      err = AX_NO_MEM;
    }
    else if (AX_SUCCESS != (err = client->init()))
    {
      MLOG(ERROR, "client init fail, err=%d", err);
    }
    else if (AX_SUCCESS != (err = client->start_group(leader, server_list)))
    {
      MLOG(ERROR, "client start_group fail, err=%d", err);
    }
    return err;
  }
  int stop_group(const char* leader_spec, const char* server_list_spec) {
    int err = AX_SUCCESS;
    AxAdminClient* client = NULL;
    Server leader;
    ServerList server_list;
    if (AX_SUCCESS != (err = parse(leader, leader_spec)))
    {
      MLOG(ERROR, "invalid leader: %s", leader_spec);
    }
    else if (AX_SUCCESS != (err = parse(server_list, server_list_spec)))
    {
      MLOG(ERROR, "invalid server_list: %s", server_list_spec);
    }
    else if (NULL == (client = get_client()))
    {
      err = AX_NO_MEM;
    }
    else if (AX_SUCCESS != (err = client->init()))
    {
      MLOG(ERROR, "client init fail, err=%d", err);
    }
    else if (AX_SUCCESS != (err = client->stop_group(leader, server_list)))
    {
      MLOG(ERROR, "client stop_group fail, err=%d", err);
    }
    return err;
  }
  int propose(const char* leader_spec, const char* cursor_spec, const char* content_spec) {
    int err = AX_SUCCESS;
    AxAdminClient* client = NULL;
    Server leader;
    Cursor cursor;
    Buffer content;
    if (AX_SUCCESS != (err = parse(leader, leader_spec)))
    {
      MLOG(ERROR, "invalid leader: %s", leader_spec);
    }
    else if (AX_SUCCESS != (err = parse(cursor, cursor_spec)))
    {
      MLOG(ERROR, "invalid cursor_spec: %s", cursor_spec);
    }
    else if (AX_SUCCESS != (err = parse(content, content_spec)))
    {
      MLOG(ERROR, "invalid content_spec: %s", content_spec);
    }
    else if (NULL == (client = get_client()))
    {
      err = AX_NO_MEM;
    }
    else if (AX_SUCCESS != (err = client->init()))
    {
      MLOG(ERROR, "client init fail, err=%d", err);
    }
    else if (AX_SUCCESS != (err = client->propose(leader, cursor, content)))
    {
      MLOG(ERROR, "client stop_group fail, err=%d", err);
    }
    return err;
  }
  int read(const char* server_spec, const char* cursor_spec) {
    int err = AX_SUCCESS;
    AxAdminClient* client = NULL;
    Server server;
    Cursor cursor;
    Buffer content;
    if (AX_SUCCESS != (err = parse(server, server_spec)))
    {
      MLOG(ERROR, "invalid server: %s", server_spec);
    }
    else if (AX_SUCCESS != (err = parse(cursor, cursor_spec)))
    {
      MLOG(ERROR, "invalid cursor_spec: %s", cursor_spec);
    }
    else if (NULL == (client = get_client()))
    {
      err = AX_NO_MEM;
    }
    else if (AX_SUCCESS != (err = client->init()))
    {
      MLOG(ERROR, "client init fail, err=%d", err);
    }
    else if (AX_SUCCESS != (err = client->read(server, cursor, content)))
    {
      MLOG(ERROR, "client stop_group fail, err=%d", err);
    }
    else
    {
      content.dump();
    }
    return err;
  }
protected:
  AxLogServer* get_server() {
    static AxLogServer server;
    return &server;
  }
  AxAdminClient* get_client() {
    static AxAdminClient client;
    return &client;
  }
};

const char* __usages__ = "Usages:\n"
  "\tax start workdir\n"
  "\tax bootstrap workdir\n"
  "\tax start_group leader_ip:leader_port ip1:port1,ip2:port2...\n"
  "\tax stop_group leader_ip:leader_port ip1:port1,ip2:port2...\n"
  "\tax propose leader_ip:leader_port term:pos log_content\n"
  "\tax read ip:port term:pos -> \n";

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
  define_cmd_call(app.bootstrap, StrArg(workdir));
  define_cmd_call(app.start, StrArg(workdir));
  define_cmd_call(app.start_group, StrArg(leader), StrArg(server_list));
  define_cmd_call(app.stop_group, StrArg(leader), StrArg(server_list));
  define_cmd_call(app.propose, StrArg(leader), StrArg(cursor), StrArg(content));
  define_cmd_call(app.read, StrArg(server), StrArg(cursor));

  if (AX_CMD_ARGS_NOT_MATCH == err)
  {
    fprintf(stderr, __usages__);
  }
  return err;
}
