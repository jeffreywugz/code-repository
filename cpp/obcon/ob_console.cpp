/**
 * (C) 2010-2011 Alibaba Group Holding Limited.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * version 2 as published by the Free Software Foundation.
 * 
 * Version: $Id$
 *
 *
 * Authors:
 *   yuanqi.xhf <yuanqi.xhf@taobao.com>
 *
 */
#include <getopt.h>
#include "ob_client2.h"
#include "common/ob_schema.h"
#include "ob_client_helper.h"
#include "common/ob_tablet_info.h"
#include "common/utility.h"
#include "ob_utils.h"
#include "utils/sh_utils.h"

const int64_t MAX_STR_LEN = 1<<12;
const int64_t MAX_N_COLUMNS = 128;
const int64_t MAX_REPLICAS_COUNT = 1<<12;
class ObMockClient: public ObBaseMockClient
{
  public:
    
    int init(const ObServer server, int64_t timeout)
  {
    return ObBaseMockClient::init(server, timeout);
  }
    
    int init_client_helper(ObClientHelper2& client_helper, const ObServer root_server)
  {
    int err = OB_SUCCESS;
    client_helper.init(get_rpc(), &get_rpc_buffer(), root_server, timeout_);
    return err;
  }
    
    int fetch_schema(ObSchemaManagerV2& schema_mgr)
  {
    return send_request(OB_FETCH_SCHEMA, dummy_, schema_mgr, timeout_);
  }

    int get_last_frozen_version(int64_t& frozen_version)
  {
    return send_request(OB_UPS_GET_LAST_FROZEN_VERSION, dummy_, frozen_version, timeout_);
  }
};    

ObServer& to_server(ObServer& server, const char *addr, int32_t port)
{
    server.set_ipv4_addr(addr, port);
    return server;
}

int desc_tables(ObDataBuffer& buf, ObSchemaManagerV2* schema_mgr, const char* table_name, const char*& result)
{
  int err = OB_SUCCESS;
  int n_column = 0;
  ObTableSchema* table_schema = NULL;
  const ObColumnSchemaV2* columns = NULL;
  const char* tmp_result = buf.get_data() + buf.get_position();
  TBSYS_LOG(INFO, "desc_tables(table_name='%s')", table_name);
  if (NULL == schema_mgr)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "schema_mgr == NULL");
  }
  else if (NULL == table_name || 0 == strcmp(table_name, "*"))
  {
    for (const ObTableSchema* it=schema_mgr->table_begin();
         OB_SUCCESS == err && it != schema_mgr->table_end(); ++it)
    {
      if (OB_SUCCESS != (err = strformat(buf, "[%ld %s] ", it->get_table_id(), it->get_table_name())))
      {
        TBSYS_LOG(ERROR, "strfomat()=>%d", err);
      }
    }
  }
  else
  {
    if (NULL == (table_schema = schema_mgr->get_table_schema(table_name)))
    {
      err = OB_SCHEMA_ERROR;
      TBSYS_LOG(ERROR, "schema_mgr->get_table_schema(table_name=%s)=>NULL", table_name);
    }
    else if (NULL == (columns = schema_mgr->get_table_schema(table_schema->get_table_id(), n_column)))
    {
      err = OB_SCHEMA_ERROR;
      TBSYS_LOG(ERROR, "schema_mgr->get_table_schema(table_name=%s)=>NULL", table_name);
    }
    for(int i = 0; OB_SUCCESS == err && i < n_column; i++)
    {
      if (OB_SUCCESS != (err = strformat(buf, "[%ld %s %s] ", columns[i].get_id(), columns[i].get_name(), obj_type_repr(columns[i].get_type()))))
      {
        TBSYS_LOG(ERROR, "strformat()=>%d", err);
      }
    }
  }
  if (OB_SUCCESS == err)
  {
    result = tmp_result;
  }
  return err;
}

int make_whole_range(ObBorderFlag& border_flag)
{
  int err = OB_SUCCESS;
  border_flag.set_min_value();
  border_flag.set_max_value();
  return err;
}

int make_version_range(ObVersionRange& version_range, int64_t start_version)
{
  int err = OB_SUCCESS;
  version_range.border_flag_.set_inclusive_start();
  version_range.border_flag_.unset_inclusive_end();
  version_range.border_flag_.unset_min_value();
  version_range.border_flag_.set_max_value();
  version_range.start_version_ = start_version;
  version_range.end_version_ = 0;
  return err;
}

int add_columns_to_scan_param(ObDataBuffer& buf, ObScanParam& scan_param, int n_columns, char** const columns)
{
  int err = OB_SUCCESS;
  ObString str;
  for (int i = 0; OB_SUCCESS == err && i < n_columns; i++)
  {
    if (OB_SUCCESS != (err = alloc_str(buf, str, columns[i])))
    {
      TBSYS_LOG(ERROR, "alloc_str(%s)=>%d", columns[i], err);
    }
    else if (OB_SUCCESS != (err = scan_param.add_column(str)))
    {
      TBSYS_LOG(ERROR, "scan_param.add_column(%.*s)=>%d", str.length(), str.ptr(), err);
    }
  }
  return err;
}

int set_range(ObScanParam& scan_param, ObString& table_name, int64_t start_version)
{
  int err = OB_SUCCESS;
  ObRange range;
  ObVersionRange version_range;
  make_whole_range(range.border_flag_);
  make_version_range(version_range, start_version);
  scan_param.set_version_range(version_range);
  scan_param.set(OB_INVALID_ID, table_name, range);
  TBSYS_LOG(DEBUG, "scan_param{table_id=%ld, table_name=%.*s}", scan_param.get_table_id(),
            scan_param.get_table_name().length(), scan_param.get_table_name().ptr());
  return err;
}

int scan_table(ObDataBuffer& buf, ObScanParam& scan_param, ObString& table_name, int64_t start_version,
                    int n_columns, char** const columns, int64_t limit)
{
  int err = OB_SUCCESS;
  scan_param.reset();
  if (OB_SUCCESS != (err = set_range(scan_param, table_name, start_version)))
  {
    TBSYS_LOG(ERROR, "set_range(table_name=%.*s, start_version=%ld)=>%d",
              table_name.length(), table_name.ptr(), start_version, err);
  }
  else if ((1 == n_columns && strcmp("*", columns[0]) || 1 < n_columns)
           && OB_SUCCESS != (err = add_columns_to_scan_param(buf, scan_param, n_columns, columns)))
  {
    TBSYS_LOG(ERROR, "add_columns_to_scan_param()=>%d", err);
  }
  else if (OB_SUCCESS != (err = scan_param.set_limit_info(0, limit)))
  {
    TBSYS_LOG(ERROR, "scan_param.set_limit_info(offset=%ld, limit=%ld)=>%d", 0, limit, err);
  }
  return err;
}

int make_tablet_range_for_create(ObRange& range, int64_t table_id)
{
  int err = OB_SUCCESS;
  range.table_id_ = table_id;
  make_whole_range(range.border_flag_);
  return err;
}

int make_tablet_info_for_create(ObTabletInfo& tablet_info, int64_t table_id)
{
  int err = OB_SUCCESS;
  make_tablet_range_for_create(tablet_info.range_, table_id);
  return err;
}

class ObConsole: public Interpreter
{
  public:
    ObConsole(): interactive(false), last_frozen_version(-1){}  
    virtual ~ObConsole();
    virtual int execute(const char* st);    
    int do_init(ObDataBuffer& buf, const char* rs_addr, const char* ups_addr, const char*& result);
    int do_desc(ObDataBuffer& buf, const char* table, const char*& result);
    int do_select(ObDataBuffer& buf, const char* table, const char* columns, const char* server, const char*& result);
    int do_create(ObDataBuffer& buf, const char* table, const char* tablet_servers, const char*& result);
    int do_report(ObDataBuffer& buf, const char* table, const char* start_key, const char* end_key,
                  const char* tablet_server, const char*& result);
    int init(ObDataBuffer& buf, const char* cmd, const char*& result);
    int desc(ObDataBuffer& buf, const char* cmd, const char*& result);
    int select(ObDataBuffer& buf, const char* cmd, const char*& result);
    int create(ObDataBuffer& buf, const char* cmd, const char*& result);
    int report(ObDataBuffer& buf, const char* cmd, const char*& result);
    int update(ObDataBuffer& buf, const char* cmd, const char*& result);
    bool interactive;    
    int64_t last_frozen_version;
    ObMockClient rs_client, ups_client;
    ObClientHelper2 client_helper;
    ObSchemaManagerV2 schema_mgr;
};

ObConsole::~ObConsole()
{
  rs_client.destroy();
  ups_client.destroy();
}

typedef int (ObConsole::*interpreter_handler_t)(ObDataBuffer& buf, const char* cmd, const char*& result);
int ObConsole::execute(const char* cmd)
{
  int err = OB_PARTIAL_FAILED;
  static char cbuf[MAX_STR_LEN];
  static ObDataBuffer buf(cbuf, sizeof(cbuf));
  const char* result = NULL;
  static interpreter_handler_t handlers[] = {
    &ObConsole::init,
    &ObConsole::desc,
    &ObConsole::select,
    &ObConsole::create,
    &ObConsole::report,
    &ObConsole::update,
  };
  static const char* handler_docs[] = {
    "init rs_addr ups_addr",
    "desc table_name",
    "select col1,col2... form table_name at ip:port",
    "create table table_name at ip1:port1,ip2:port2,ip3:port3",
    "report tablet table_name[start,end) at ip:port",
    "update table_name set col1=val1,col2=val2... where rowkey=rowkey",
    "help",
  };
  TBSYS_LOG(INFO, "ObConsole::execute('%s')", cmd);
  buf.get_position() = 0;
  if (NULL == cmd)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "execute(cmd=NULL)");
  }
  for (int i = 0; OB_PARTIAL_FAILED == err && i < (int)ARRAYSIZEOF(handlers); i++)
  {
    TBSYS_LOG(INFO, "execute(cmd='%s', try='%s')", cmd, handler_docs[i]);
    if (OB_SUCCESS != (err = (this->*handlers[i])(buf, cmd, result)) && OB_PARTIAL_FAILED != err)
    {
      TBSYS_LOG(ERROR, "handlers[%d]('%s', doc='%s')=>%d", i, cmd, handler_docs[i], err);
    }
  }
  if (!interactive)
  {}
  else if(OB_SUCCESS == err)
  {
    fprintf(stdout, "%s\n", result);
  }
  else if(OB_PARTIAL_FAILED == err)
  {
    fprintf(stderr, "ObConsole::execute(cmd='%s')=>'No Matching Command'\n", cmd);
    fprintf(stderr, "Available Command:\n");
    for (int i = 0; i < (int)ARRAYSIZEOF(handler_docs); i++)
    {
      fprintf(stderr, "\t%s\n", handler_docs[i]);
    }
  }
  else
  {
    fprintf(stderr, "ObConsole::execute(cmd='%s')=>%d\n", cmd, err);
  }
  return err;
}

int ObConsole::init(ObDataBuffer& buf, const char* cmd, const char*& result)
{
  int err = OB_SUCCESS;
  const char* pat = "init +([a-zA-Z0-9._:]+) ([a-zA-Z0-9._:]+)";
  char* rs_addr = NULL;
  char* ups_addr = NULL;
  (OB_SUCCESS == (err = reg_parse2(pat, cmd, buf, &rs_addr, &ups_addr)))
    && (OB_SUCCESS == (err = do_init(buf, rs_addr, ups_addr, result)));
  return err;
}

int ObConsole::desc(ObDataBuffer& buf, const char* cmd, const char*& result)
{
  int err = OB_SUCCESS;
  const char* pat = "desc +([0-9A-Za-z_*]*)";
  const char* table = NULL;
  (OB_SUCCESS == (err = reg_parse2(pat, cmd, buf, &table)))
    && (OB_SUCCESS == (err = do_desc(buf, table, result)));
  return err;
}

int ObConsole::select(ObDataBuffer& buf, const char* cmd, const char*& result)
{
  int err = OB_SUCCESS;
  const char* pat = "select +(.+) +from +(.*) +at +(.*)";
  const char* columns = NULL;
  const char* table = NULL;
  const char* server = NULL;
  (OB_SUCCESS == (err = reg_parse2(pat, cmd, buf, &columns, &table, &server)))
    && (OB_SUCCESS == (err = do_select(buf, table, columns, server, result)));
  return err;
}

int ObConsole::create(ObDataBuffer& buf, const char* cmd, const char*& result)
{
  int err = OB_SUCCESS;
  const char* pat = "create +table +([a-zA-Z0-9_]+) +at +([,.a-zA-Z0-9_:]+)";
  const char* table = NULL;
  const char* servers = NULL;
  (OB_SUCCESS == (err = reg_parse2(pat, cmd, buf, &table, &servers)))
    && (OB_SUCCESS == (err = do_create(buf, table, servers, result)));
  return err;
}

int ObConsole::report(ObDataBuffer& buf, const char* cmd, const char*& result)
{
  int err = OB_SUCCESS;
  const char* pat = "report +tablet +(.+)\[(.+),(.+)[)] +at +(.*)";
  const char* table_name = NULL;
  const char* start_key = NULL;
  const char* end_key = NULL;
  const char* server = NULL;
  (OB_SUCCESS == (err = reg_parse2(pat, cmd, buf, &table_name, &start_key, &end_key, &server)))
    && (OB_SUCCESS == (err = do_report(buf, table_name, start_key, end_key, server, result)));
  return err;
}

int ObConsole::update(ObDataBuffer& buf, const char* cmd, const char*& result)
{
  int err = OB_SUCCESS;
  UNUSED(buf);
  UNUSED(cmd);
  UNUSED(result);
  err = OB_PARTIAL_FAILED;
  return err;
}

int ObConsole::do_init(ObDataBuffer& buf, const char* rs_addr, const char* ups_addr, const char*& result)
{
  int err = OB_SUCCESS;
  int64_t timeout = 1000 * 1000;
  ObServer root_server;
  ObServer update_server;
  UNUSED(buf);
  TBSYS_LOG(INFO, "do_init(rs_addr='%s', ups_addr='%s')", rs_addr, ups_addr);
  if (OB_SUCCESS != (err = to_server(root_server, rs_addr))
      || OB_SUCCESS != (err = to_server(update_server, ups_addr)))
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "init(rs_addr=%s, ups_addr=%s)=>%d", rs_addr, ups_addr, err);
  }
  else if (OB_SUCCESS != (err = rs_client.init(root_server, timeout)))
  {
    TBSYS_LOG(ERROR, "client.init()=>%d", err);
  }
  else if (OB_SUCCESS != (err = ups_client.init(update_server, timeout)))
  {
    TBSYS_LOG(ERROR, "client.init()=>%d", err);
  }
  else if (OB_SUCCESS != (err = rs_client.init_client_helper(client_helper, root_server)))
  {
    TBSYS_LOG(ERROR, "init_client_helper()=>%d", err);
  }
  else if (OB_SUCCESS != (err = rs_client.fetch_schema(schema_mgr)))
  {
    TBSYS_LOG(ERROR, "fetch_schema()=>%d", err);
  }
  else if (OB_SUCCESS != (err = ups_client.get_last_frozen_version(last_frozen_version)))
  {
    TBSYS_LOG(ERROR, "get_last_frozen_version()=>%d", err);
  }
  else
  {
    result = "init OK.";
  }
  return err;
}

int ObConsole::do_desc(ObDataBuffer& buf, const char* table, const char*& result)
{
  return desc_tables(buf, &schema_mgr, table, result);
}

int ObConsole::do_select(ObDataBuffer& buf, const char* table, const char* columns_spec,
                                  const char* server_spec, const char*& result)
{
  int err = OB_SUCCESS;
  int n_columns = 0;
  char* columns[MAX_N_COLUMNS];
  ObTableSchema* table_schema = NULL;
  ObString table_name;
  ObServer server;
  int64_t start_version = last_frozen_version + 1;
  ObScanParam scan_param;
  ObScanner scanner;
  char* tmp_pos = NULL;
  int64_t limit = 10;
  TBSYS_LOG(INFO, "select(table='%s', columns='%s' ,server='%s')", table, columns_spec, server_spec);
  if (NULL == table || NULL == columns_spec || NULL == server_spec)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "select(table=%s, columns=%s, spec=%s)=>%d", table, columns_spec, server_spec, err);
  }
  else if (OB_SUCCESS != (err = to_server(server, server_spec)))
  {
    TBSYS_LOG(ERROR, "to_server(spec=%s)=>%d", server_spec, err);
  }
  else if (NULL == (table_schema = schema_mgr.get_table_schema(table)))
  {
    err = OB_SCHEMA_ERROR;
    TBSYS_LOG(ERROR, "schema_mgr.get_table_schema(table=%s)=>%d", table, err);
  }
  else if (OB_SUCCESS != (err = split(buf, columns_spec, ", ", ARRAYSIZEOF(columns), n_columns, columns)))
  {
    TBSYS_LOG(ERROR, "split(column_spec=%s)=>%d", columns_spec, err);
  }
  else if (OB_SUCCESS != (err = alloc_str(buf, table_name, table)))
  {
    TBSYS_LOG(ERROR, "alloc_str(table=%s)=>%d", table, err);
  }
  else if (OB_SUCCESS != (err = scan_table(buf, scan_param, table_name, start_version, n_columns, columns, limit)))
  {
    TBSYS_LOG(ERROR, "scan_table(table_schema=%p, columns=%s)=>%d", table_schema, columns_spec, err);
  }
  else if (OB_SUCCESS != (err = client_helper.scan(server, scan_param, scanner)))
  {
    TBSYS_LOG(ERROR, "client_helper.scan(table=%s)=>%d", table, err);
  }
  else
  {
    tmp_pos = buf.get_data() + buf.get_position();
  }
  
  if (OB_SUCCESS != err)
  {}
  else if (OB_SUCCESS != (err = repr(buf, scanner)))
  {
    TBSYS_LOG(ERROR, "scanner_format()=>%d", err);
  }
  else
  {
    result = tmp_pos;
  }
  return err;
}

int ObConsole::do_create(ObDataBuffer& buf, const char* table_name, const char* tablet_servers, const char*& result)
{
  int err = OB_SUCCESS;
  //int64_t tablet_version = 0;
  int replica_num = 0;
  ObTabletInfo tablet_info;
  ObTableSchema* table_schema = NULL;
  ObServer servers[MAX_REPLICAS_COUNT];
  if (NULL == table_name || NULL == tablet_servers)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "create_table(table_name=%p, tablet_servers=%s)=>%d", table_name, tablet_servers, err);
  }
  else if (NULL == (table_schema = schema_mgr.get_table_schema(table_name)))
  {
    err = OB_SCHEMA_ERROR;
    TBSYS_LOG(ERROR, "schema_mgr.get_table_schema(%s)=>%d", table_name, err);
  }
  else if (OB_SUCCESS != (err = make_tablet_info_for_create(tablet_info, table_schema->get_table_id())))
  {
    TBSYS_LOG(ERROR, "make_tablet_info_for_create(table_id=%s)=>%d", table_schema->get_table_id(), err);
  }
  else if (OB_SUCCESS != (err = parse_servers(tablet_servers, ARRAYSIZEOF(servers), replica_num, servers)))
  {
    TBSYS_LOG(ERROR, "parse_servers(tablet_servrs=%s)=>%d", tablet_servers, err);
  }
  // else if (OB_SUCCESS != (err = rt_mgr.create_table(*table_schema, tablet_info, replica_num, servers, tablet_version)))
  // {
  //   TBSYS_LOG(ERROR, "rt_mgr.create_table(tablet_servrs=%s, replca_num=%d)=>%d", tablet_servers, replica_num, err);
  // }
  return err;
}

int ObConsole::do_report(ObDataBuffer& buf, const char* table_name, const char* start_key, const char* end_key,
                                  const char* server, const char*& result)
{
  int err = OB_SUCCESS;
  ObTableSchema* table_schema = NULL;
  ObServer _server;
  if (NULL == table_name || NULL == server)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "do_report(table_name=%p, server=%s)=>%d", table_name, server, err);
  }
  else if (NULL == (table_schema = schema_mgr.get_table_schema(table_name)))
  {
    err = OB_SCHEMA_ERROR;
    TBSYS_LOG(ERROR, "schema_mgr.get_table_schema(%s)=>%d", table_name, err);
  }
  else if (OB_SUCCESS != (err = to_server(_server, server)))
  {
    TBSYS_LOG(ERROR, "to_server(tablet_servrs=%s)=>%d", server, err);
  }
  // else if (OB_SUCCESS != (err = rt_mgr.report_tablet(*table_schema, tablet_info, _server, tablet_version)))
  // {
  //   TBSYS_LOG(ERROR, "rt_mgr.create_table(tablet_servr=%s, start_key=%s, end_key=%s)=>%d", server, start_key, end_key, err);
  // }
  return err;
}

struct Config
{
  Config(): interactive(false), keep_going(false), log_file(NULL), init_cmd(NULL), init_file(NULL)
  {}
  ~Config()
  {}
  bool interactive;
  bool keep_going;
  const char* log_file;
  const char* init_cmd;
  const char* init_file;
  int dump()
  {
    TBSYS_LOG(INFO, "config{interactive=%s, keep_going=%s, log_file='%s', init_cmd='%s', init_file='%s'}", 
              STR_BOOL(interactive), STR_BOOL(keep_going), log_file, init_cmd, init_file);
    return OB_SUCCESS;
  }
  int parse_argv(int argc, char** argv)
  {
    int err = OB_SUCCESS;
    int opt = 0;
    const char* usage = "Usages:\n"
      "\t%1$s -h # show this help\n"
      "\t%1$s -c 'cmd' # execute 'cmd'\n"
      "\t%1$s -f 'script-file' # execute cmd in 'script-file'\n"
      "\t%1$s -i # execute cmd interactivly\n"
      "Common Options:\n"
      "\t-k keep going on error\n"
      "\t-l log-file\n"
      ;
    const char* opt_string = "c:f:l:kih";
    struct option longopts[] = {
      {"cmd", 1, NULL, 'c'},
      {"file", 1, NULL, 'f'},
      {"log", 1, NULL, 'l'},
      {"keep-goine", 0, NULL, 'k'},
      {"interact", 0, NULL, 'i'},
      {"help", 0, NULL, 'h'},
    };
    while (-1 != (opt = getopt_long(argc, argv, opt_string, longopts, NULL)))
    {
      switch (opt)
      {
        case 'k':
          keep_going = true;
          break;
        case 'i':
          interactive = true;
          break;
        case 'c':
          init_cmd = optarg;
          break;
        case 'f':
          init_file = optarg;
          break;
        case 'l':
          log_file = optarg;
          break;
        case 'h':
        default:
          err = OB_INVALID_ARGUMENT;
          break;
      }
    }
    if (OB_SUCCESS == err && (!interactive || NULL == init_cmd || NULL == init_file))
    {
      err = OB_INVALID_ARGUMENT;
    }
    if (OB_SUCCESS != err)
    {
      fprintf(stderr, usage, argv[0]);
    }
    return err;
  }
};

ObConsole* __rt_con = NULL;
int do_main(Config& cfg)
{
  int err = OB_SUCCESS;
  static const char* delim = ";\n";
  static const char* prompt = "> ";
  static ObConsole rt_con;
  rt_con.interactive = cfg.interactive;
  __rt_con = &rt_con;
  cfg.dump();
  if ((cfg.keep_going || OB_SUCCESS == err)
      && cfg.init_cmd && OB_SUCCESS != (err = execute_str(&rt_con, cfg.init_cmd, cfg.keep_going, delim)))
  {
    TBSYS_LOG(ERROR, "execute_str(rt_con, cmd='%s')=>%d", cfg.init_cmd, err);
  }
  if ((cfg.keep_going || OB_SUCCESS == err)
      && cfg.init_file && OB_SUCCESS != (err = execute_file(&rt_con, cfg.init_file, cfg.keep_going, delim)))
  {
    TBSYS_LOG(ERROR, "execute_file(rt_con, file='%s')=>%d", err);
  }
  
  if ((cfg.keep_going || OB_SUCCESS == err)
      && cfg.interactive && OB_SUCCESS != (err = execute_interactive(&rt_con, cfg.keep_going, delim, prompt)))
  {
    TBSYS_LOG(ERROR, "execute_interactive(rt_con)=>%d", err);
  }
  return err;
}

int do_cmd(const char* cmd)
{
  return execute_str(__rt_con, cmd, false, ";\n");
}

int main(int argc, char** argv)
{
  int err = OB_SUCCESS;
  int do_main_err = OB_SUCCESS;
  Config config;
  if (OB_SUCCESS != (err = config.parse_argv(argc, argv)))
  {
    if (OB_INVALID_ARGUMENT != err)
    {
      TBSYS_LOG(ERROR, "config.parse_argv(argc, argv)=>%d", err);
    }
  }
  else if (OB_SUCCESS != (err = ob_init_memory_pool()))
  {
    TBSYS_LOG(ERROR, "ob_init_memory_pool()=>%d", err);
  }
  else if (NULL != config.log_file)
  {
    TBSYS_LOG(INFO, "log_file=%s", config.log_file);
    TBSYS_LOGGER.setFileName(config.log_file, true);
  }
  
  if (OB_SUCCESS != err)
  {}
  else if (OB_SUCCESS != (do_main_err = do_main(config)))
  {
    TBSYS_LOG(ERROR, "do_main()=>%d");
  }
  
  return OB_SUCCESS == err? do_main_err: err;
}
