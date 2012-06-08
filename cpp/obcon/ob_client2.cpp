/**
 * (C) 2007-2010 Taobao Inc.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * Version: $Id$
 *
 * Authors:
 *   yuanqi <yuanqi.xhf@taobao.com>
 *     - some work details if you want
 */

#include "stdlib.h"
#include "tbsys.h"
#include "common/ob_malloc.h"
#include "common/ob_obi_role.h"
#include "common/ob_schema.h"
#include "common/ob_client_config.h"
#include "common/ob_log_cursor.h"
#include "common/ob_scan_param.h"
#include "common/ob_mutator.h"
#include "updateserver/ob_ups_mutator.h"
#include "ob_utils.h"
#include "ob_client2.h"
#include "file_utils.h"
#include "updateserver/ob_log_sync_delay_stat.h"
#include "updateserver/ob_ups_clog_status.h"

using namespace oceanbase::common;
using namespace oceanbase::updateserver;

const int64_t MAX_STR_LEN = 1<<14;
const int64_t MAX_N_COLUMNS = 1<<12;

const char* _usages = "Usages:\n"
  "\t# You can set env var 'log_level' to 'DEBUG'/'WARN'...\n"
  "\t%1$s desc rs_ip:rs_port table\n"
  "\t%1$s get_obi_role rs_ip:rs_port\n"
  "\t%1$s get_master_ups rs_ip:rs_port\n"
  "\t%1$s get_client_cfg rs_ip:rs_port\n"
  "\t%1$s get_ms rs_ip:rs_port\n"
  "\t%1$s get_replayed_cursor ups_ip:ups_port\n"
  "\t%1$s get_max_log_seq_replayable ups_ip:ups_port\n"
  "\t%1$s get_last_frozen_version ups_ip:ups_port\n"
  "\t%1$s scan rs_ip:rs_port table columns rowkey limit server\n"
  "\t# default scan args: columns='*', rowkey=[min,max], limit=10, server=ms\n"
  "\t%1$s set rs_ip:rs_port table column rowkey value server\n"
  "\t# default set args: server=ms\n"
  "\t%1$s send_mutator rs log_file server=ups\n"
  "\t%1$s get_log_sync_delay_stat ups\n"
  "\t%1$s get_clog_status ups\n";

struct ServerList
{
  static const int64_t MAX_N_SERVERS = 1<<5;
  int32_t n_servers_;
  ObServer servers_[MAX_N_SERVERS];
  ServerList(): n_servers_(0) {}
  ~ServerList(){}
  int serialize(char* buf, int64_t len, int64_t& pos) const
  {
    return OB_NOT_SUPPORTED;
  }
  int deserialize(char* buf, int64_t len, int64_t& pos)
  {
    int err = OB_SUCCESS;
    int64_t reserved = 0;
    if (OB_SUCCESS != (err = serialization::decode_vi32(buf, len, pos, &n_servers_)))
    {
      TBSYS_LOG(ERROR, "deserialize server_num error, err=%d", err);
    }
    for(int64_t i = 0; OB_SUCCESS == err && i < n_servers_; i++)
    {
      if (OB_SUCCESS != (err = servers_[i].deserialize(buf, len, pos)))
      {
        TBSYS_LOG(ERROR, "deserialize %ldth SERVER error, ret=%d", i, err);
      }
      else if (OB_SUCCESS != (err = serialization::decode_vi64(buf, len, pos, &reserved)))
      {
        TBSYS_LOG(ERROR, "deserialize reserve field error, ret=%d", err);
      }
    }
    return err;
  }
  char* to_str(char* buf, int64_t len, int64_t& pos)
  {
    int err = OB_SUCCESS;
    char* ret_str = buf + pos;
    if (OB_SUCCESS != (err = strformat(buf, len, pos, "n_server=%d ", n_servers_)))
    {
      TBSYS_LOG(ERROR, "strformat()=>%d", err);
    }
    for(int64_t i = 0; i < n_servers_; i++)
    {
      if (OB_SUCCESS != (err = strformat(buf, len, pos, "%s ", servers_[i].to_cstring())))
      {
        TBSYS_LOG(ERROR, "strformat()=>%d", err);
      }
    }
    if (OB_SUCCESS != err)
    {
      ret_str = NULL;
    }
    return ret_str;
  }
};

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

int set_range(ObDataBuffer& buf, ObScanParam& scan_param, ObString& table_name, int64_t start_version, const char* start_key, const char* end_key)
{
  int err = OB_SUCCESS;
  ObRange range;
  ObVersionRange version_range;
  if (0 == strcmp(start_key, "min"))
  {
    TBSYS_LOG(INFO, "start_key is min");
    range.border_flag_.set_min_value();
  }
  else if (OB_SUCCESS != (err = hex2bin(buf, range.start_key_, start_key, strlen(start_key))))
  {
    TBSYS_LOG(ERROR, "hex2bin()=>%d", err);
  }
  if (0 == strcmp(end_key, "max"))
  {
    TBSYS_LOG(INFO, "end_key is max");
    range.border_flag_.set_max_value();
  }
  else if (OB_SUCCESS != (err = hex2bin(buf, range.end_key_, end_key, strlen(end_key))))
  {
    TBSYS_LOG(ERROR, "hex2bin()=>%d", err);
  }
  range.border_flag_.set_inclusive_start();
  range.border_flag_.set_inclusive_end();
  make_version_range(version_range, start_version);
  scan_param.set_version_range(version_range);
  scan_param.set(OB_INVALID_ID, table_name, range);
  TBSYS_LOG(DEBUG, "scan_param{table_id=%ld, table_name=%.*s}", scan_param.get_table_id(),
            scan_param.get_table_name().length(), scan_param.get_table_name().ptr());
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

int scan_func(ObDataBuffer& buf, ObScanParam& scan_param, int64_t start_version, const char* _table_name,
               int n_columns, char** const columns, char* start_key, char* end_key, int64_t limit)
{
  int err = OB_SUCCESS;
  ObString table_name;
  scan_param.reset();
  if (OB_SUCCESS != (err = alloc_str(buf, table_name, _table_name)))
  {
    TBSYS_LOG(ERROR, "alloc_str(%s)=>%d", _table_name, err);
  }
  else if (OB_SUCCESS != (err = set_range(buf, scan_param, table_name, start_version, start_key, end_key)))
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
    TBSYS_LOG(ERROR, "scan_param.set_limit_info(offset=%d, limit=%ld)=>%d", 0, limit, err);
  }
  return err;
}

int scan_func2(ObDataBuffer& buf, ObScanParam& scan_param, int64_t start_version,
               const char* table, const char* columns_spec, const char* rowkey_range, int64_t limit)
{
  int err = OB_SUCCESS;
  int n_columns = 0;
  char* columns[MAX_N_COLUMNS];
  char* start_key = NULL;
  char* end_key = NULL;
  TBSYS_LOG(INFO, "scan(start_version=%ld, table='%s', columns='%s', rowkey='%s')", start_version, table, columns_spec, rowkey_range);
  if (NULL == table || NULL == columns_spec || NULL == rowkey_range)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "select(table=%s, columns=%s, rowkey=%s)=>%d", table, columns_spec, rowkey_range, err);
  }
  else if (OB_SUCCESS != (err = split(buf, columns_spec, ", ", ARRAYSIZEOF(columns), n_columns, columns)))
  {
    TBSYS_LOG(ERROR, "split(column_spec=%s)=>%d", columns_spec, err);
  }
  else if (OB_SUCCESS != (err = parse_range(buf, rowkey_range, start_key, end_key)))
  {
    TBSYS_LOG(ERROR, "parse_range(%s)=>%d", rowkey_range, err);
  }
  else if (OB_SUCCESS != (err = scan_func(buf, scan_param, start_version, table, n_columns, columns, start_key, end_key, limit)))
  {
    TBSYS_LOG(ERROR, "scan_table(table_schema=%s, columns=%s, rowkey=[%s,%s])=>%d", table, columns_spec, start_key, end_key, err);
  }
  return err;
}

int set_obj(ObDataBuffer& buf, ObObj& obj, ColumnType type, const char* value)
{
  int err = OB_SUCCESS;
  ObString str_value;
  switch(type)
  {
    case ObIntType:
      obj.set_int(atoll(value));
      break;
    case ObBoolType:
      obj.set_bool(atoi(value));
      break;
    case ObVarcharType:
      if (OB_SUCCESS != (err = alloc_str(buf, str_value, value)))
      {
        TBSYS_LOG(ERROR, "alloc_str(%s)=>%d", value, err);
      }
      else
      {
        obj.set_varchar(str_value);
      }
      break;
    case ObFloatType:
      obj.set_float(atof(value));
      break;
    case ObDoubleType:
      obj.set_double(atof(value));
      break;
    case ObDateTimeType:
      obj.set_datetime(atoll(value));
      break;
    case ObPreciseDateTimeType:
      obj.set_precise_datetime(atoll(value));
      break;
    case ObCreateTimeType:
    case ObModifyTimeType:
    default:
      err = OB_NOT_SUPPORTED;
      break;
  }
  return err;
}

int mutate_func(ObDataBuffer& buf, ObMutator& mutator, ObSchemaManagerV2& schema_mgr,
                const char* _table_name, const char* _rowkey, const char* _column_name, const char* _value)
{
  int err = OB_SUCCESS;
  ObString table_name;
  ObString rowkey;
  ObString column_name;
  const ObColumnSchemaV2* column_schema = NULL;
  ObString value;
  ObObj cell_value;
  mutator.reset();
  if (OB_SUCCESS != (err = alloc_str(buf, table_name, _table_name)))
  {
    TBSYS_LOG(ERROR, "alloc_str(%s)=>%d", _table_name, err);
  }
  else if (OB_SUCCESS != (err = hex2bin(buf, rowkey, _rowkey, strlen(_rowkey))))
  {
    TBSYS_LOG(ERROR, "hex2bin()=>%d", err);
  }
  else if (OB_SUCCESS != (err = alloc_str(buf, column_name, _column_name)))
  {
    TBSYS_LOG(ERROR, "alloc_str(%s)=>%d", _column_name, err);
  }
  else if (NULL == (column_schema = schema_mgr.get_column_schema(_table_name, _column_name)))
  {
    err = OB_SCHEMA_ERROR;
    TBSYS_LOG(ERROR, "schema_mgr.get_column_schema(table=%s, column=%s): NO SCHEMA FOUND",
              _table_name, _column_name);
  }
  else if (OB_SUCCESS != (err = set_obj(buf, cell_value, column_schema->get_type(), _value)))
  {
    TBSYS_LOG(ERROR, "set_obj()=>%d", err);
  }
  else if (OB_SUCCESS != (err = mutator.update(table_name, rowkey, column_name, cell_value)))
  {
    TBSYS_LOG(ERROR, "mutator.update()=>%d", err);
  }
  return err;
}

int cell_info_resolve_table_name(ObSchemaManagerV2& sch_mgr, ObCellInfo& cell)
{
  int err = OB_SUCCESS;
  uint64_t table_id = cell.table_id_;
  const ObTableSchema* table_schema = NULL;
  const char* table_name = NULL;
  // `table_id == OB_INVALID_ID' is possible when cell.op_type == OB_USE_OB or cell.op_type == OB_USE_DB
  if (OB_INVALID_ID != table_id)
  {
    if (NULL == (table_schema = sch_mgr.get_table_schema(table_id)))
    {
      err = OB_SCHEMA_ERROR;
      TBSYS_LOG(WARN, "sch_mge.get_table_schema(table_id=%lu)=>NULL", table_id);
    }
    else if (NULL == (table_name = table_schema->get_table_name()))
    {
      err = OB_SCHEMA_ERROR;
      TBSYS_LOG(ERROR, "get_table_name(table_id=%lu) == NULL", table_id);
    }
    else
    {
      cell.table_name_.assign_ptr((char*)table_name, static_cast<int32_t>(strlen(table_name)));
      //cell.table_id_ = OB_INVALID_ID;
    }
  }
  return err;
}

int cell_info_resolve_column_name(ObSchemaManagerV2& sch_mgr, ObCellInfo& cell)
{
  int err = OB_SUCCESS;
  uint64_t table_id = cell.table_id_;
  uint64_t column_id = cell.column_id_;
  const ObColumnSchemaV2* column_schema = NULL;
  const char* column_name = NULL;
  // `table_id == OB_INVALID_ID' is possible when cell.op_type == OB_USE_OB or cell.op_type == OB_USE_DB
  // `column_id == OB_INVALID_ID' is possible when cell.op_type == OB_USE_OB or cell.op_type == OB_USE_DB
  //                                                        or cell.op_type == OB_DEL_ROW
  if (OB_INVALID_ID != table_id && OB_INVALID_ID != column_id)
  {
    if (NULL == (column_schema = sch_mgr.get_column_schema(table_id, column_id)))
    {
      err = OB_SCHEMA_ERROR;
      TBSYS_LOG(ERROR, "sch_mgr.get_column_schema(table_id=%lu, column_id=%lu) == NULL", table_id, column_id);
    }
    else if(NULL == (column_name = column_schema->get_name()))
    {
      err = OB_SCHEMA_ERROR;
      TBSYS_LOG(ERROR, "get_column_name(table_id=%lu, column_id=%lu) == NULL", table_id, column_id);
    }
    else
    {
      cell.column_name_.assign_ptr((char*)column_name, static_cast<int32_t>(strlen(column_name)));
      //cell.column_id_ = OB_INVALID_ID;
    }
  }
  return err;
}

static void dump_ob_mutator_cell(ObMutatorCellInfo& cell)
{
  uint64_t op = cell.op_type.get_ext();
  uint64_t table_id = cell.cell_info.table_id_;
  uint64_t column_id = cell.cell_info.column_id_;
  ObString table_name = cell.cell_info.table_name_;
  ObString column_name = cell.cell_info.column_name_;
  TBSYS_LOG(INFO, "cell{op=%lu, table=%lu[%*s], column=%lu[%*s]", op,
            table_id, table_name.length(), table_name.ptr(), column_id, column_name.length(), column_name.ptr());
}

int dump_ob_mutator(ObMutator& mut)
{
  int err = OB_SUCCESS;
  TBSYS_LOG(DEBUG, "dump_ob_mutator");
  mut.reset_iter();
  while (OB_SUCCESS == err && OB_SUCCESS == (err = mut.next_cell()))
  {
    ObMutatorCellInfo* cell = NULL;
    if (OB_SUCCESS != (err = mut.get_cell(&cell)))
    {
      TBSYS_LOG(ERROR, "mut.get_cell()=>%d", err);
    }
    else
    {
      dump_ob_mutator_cell(*cell);
    }
  }
  if (OB_ITER_END == err)
  {
    err = OB_SUCCESS;
  }
  return err;
}

int ob_mutator_resolve_name(ObSchemaManagerV2& sch_mgr, ObMutator& mut)
{
  int err = OB_SUCCESS;
  while (OB_SUCCESS == err && OB_SUCCESS == (err = mut.next_cell()))
  {
    ObMutatorCellInfo* cell = NULL;
    if (OB_SUCCESS != (err = mut.get_cell(&cell)))
    {
      TBSYS_LOG(ERROR, "mut.get_cell()=>%d", err);
    }
    else if (OB_SUCCESS != (err = cell_info_resolve_column_name(sch_mgr, cell->cell_info)))
    {
      TBSYS_LOG(ERROR, "resolve_column_name(table_id=%lu, column_id=%lu)=>%d",
                cell->cell_info.table_id_, cell->cell_info.column_id_, err);
    }
    else if (OB_SUCCESS != (err = cell_info_resolve_table_name(sch_mgr, cell->cell_info)))
    {
      TBSYS_LOG(ERROR, "resolve_table_name(table_id=%lu)=>%d", cell->cell_info.table_id_, err);
    }
  }
  if (OB_ITER_END == err)
  {
    err = OB_SUCCESS;
  }
  return err;
}
int mutator_add_(ObMutator& dst, ObMutator& src)
{
  int err = OB_SUCCESS;
  src.reset_iter();
  while ((OB_SUCCESS == err) && (OB_SUCCESS == (err = src.next_cell())))
  {
    ObMutatorCellInfo* cell = NULL;
    if (OB_SUCCESS != (err = src.get_cell(&cell)))
    {
      TBSYS_LOG(ERROR, "mut.get_cell()=>%d", err);
    }
    else if (OB_SUCCESS != (err = dst.add_cell(*cell)))
    {
      TBSYS_LOG(ERROR, "dst.add_cell()=>%d", err);
    }
  }
  if (OB_ITER_END == err)
  {
    err = OB_SUCCESS;
  }
  return err;
}
    
int mutator_add(ObMutator& dst, ObMutator& src, int64_t size_limit)
{
  int err = OB_SUCCESS;
  if (dst.get_serialize_size() + src.get_serialize_size() > size_limit)
  {
    err = OB_SIZE_OVERFLOW;
    TBSYS_LOG(DEBUG, "mutator_add(): size overflow");
  }
  else if (OB_SUCCESS != (err = mutator_add_(dst, src)))
  {
    TBSYS_LOG(ERROR, "mutator_add()=>%d", err);
  }
  return err;
}

struct RPC : public BaseClient
{
  RPC(): is_init_(false) {}
  ~RPC() { if (is_init_)destroy(); }
  int initialize() {
    int err = OB_SUCCESS;
    if (OB_SUCCESS == (err = BaseClient::initialize()))
    {
      is_init_ = true;
    }
    return err;
  }
  int desc(const char* rs, const char* table)
  {
    int err = OB_SUCCESS;
    char cbuf[MAX_STR_LEN];
    ObDataBuffer buf(cbuf, sizeof(cbuf));
    const char* result = NULL;
    ObSchemaManagerV2 schema_mgr;
    if (OB_SUCCESS != (err = send_request(rs, OB_FETCH_SCHEMA, _dummy_, schema_mgr)))
    {
      TBSYS_LOG(ERROR, "send_request()=>%d", err);
    }
    else if (OB_SUCCESS != (err = desc_tables(buf, &schema_mgr, table, result)))
    {
      TBSYS_LOG(ERROR, "desc_table()=>%d", err);
    }
    else
    {
      printf("%s\n", result);
    }
    return err;
  }

  int get_obi_role(const char* rs)
  {
    int err = OB_SUCCESS;
    ObiRole obi_role;
    if (OB_SUCCESS != (err = send_request(rs, OB_GET_OBI_ROLE, _dummy_, obi_role)))
    {
      TBSYS_LOG(ERROR, "send_request()=>%d", err);
    }
    else
    {
      printf("obi_role=%s\n", obi_role.get_role_str());
    }
    return err;
  }

  int get_master_ups(const char* rs)
  {
    int err = OB_SUCCESS;
    ObServer ups;
    if (OB_SUCCESS != (err = send_request(rs, OB_GET_UPDATE_SERVER_INFO, _dummy_, ups)))
    {
      TBSYS_LOG(ERROR, "send_request()=>%d", err);
    }
    else
    {
      printf("%s\n", ups.to_cstring());
    }
    return err;
  }

  int get_client_cfg(const char* rs)
  {
    int err = OB_SUCCESS;
    char cbuf[MAX_STR_LEN];
    int64_t pos = 0;
    ObClientConfig clicfg;
    if (OB_SUCCESS != (err = send_request(rs, OB_GET_CLIENT_CONFIG, _dummy_, clicfg)))
    {
      TBSYS_LOG(ERROR, "send_request()=>%d", err);
    }
    else
    {
      clicfg.print(cbuf, sizeof(cbuf), pos);
      cbuf[sizeof(cbuf)-1] = 0;
      printf("%s\n", cbuf);
    }
    return err;
  }

  int get_ms(const char* rs)
  {
    int err = OB_SUCCESS;
    ServerList ms_list;
    int64_t pos = 0;
    char cbuf[MAX_STR_LEN];
    char* result = NULL;
    if (OB_SUCCESS != (err = send_request(rs, OB_GET_MS_LIST, _dummy_, ms_list)))
    {
      TBSYS_LOG(ERROR, "send_request(GET_MS_LIST)=>%d", err);
    }
    else if (NULL == (result = ms_list.to_str(cbuf, sizeof(cbuf), pos)))
    {
      TBSYS_LOG(ERROR, "ms_list.to_str()=>%d", err);
    }
    else
    {
      printf("%s\n", result);
    }
    return err;
  }

  int get_replayed_cursor(const char* ups)
  {
    int err = OB_SUCCESS;
    ObLogCursor log_cursor;
    if (OB_SUCCESS != (err = send_request(ups, OB_GET_CLOG_CURSOR, _dummy_, log_cursor)))
    {
      TBSYS_LOG(ERROR, "send_request()=>%d", err);
    }
    else
    {
      printf("%s\n", log_cursor.to_str());
    }
    return err;
  }

  int get_max_log_seq_replayable(const char* ups)
  {
    int err = OB_SUCCESS;
    int64_t log_seq = 0;
    if (OB_SUCCESS != (err = send_request(ups, OB_RS_GET_MAX_LOG_SEQ, _dummy_, log_seq)))
    {
      TBSYS_LOG(ERROR, "send_request()=>%d", err);
    }
    else
    {
      printf("%ld\n", log_seq);
    }
    return err;
  }

  int get_last_frozen_version(const char* ups)
  {
    int err = OB_SUCCESS;
    int64_t frozen_version = 0;
    if (OB_SUCCESS != (err = send_request(ups, OB_UPS_GET_LAST_FROZEN_VERSION, _dummy_, frozen_version)))
    {
      TBSYS_LOG(ERROR, "send_request()=>%d", err);
    }
    else
    {
      printf("%ld\n", frozen_version);
    }
    return err;
  }

  int choose_server(const char* rs, const char* _server, ObServer& server)
  {
    int err = OB_SUCCESS;
    ServerList ms_list;
    if (0 != strcmp("ms", _server) && 0 != strcmp("ups", _server))
    {
      if (OB_SUCCESS != (err = to_server(server, _server)))
      {
        TBSYS_LOG(ERROR, "to_server(%s)=>%d", _server, err);
      }
    }
    else if (0 == strcmp("ms", _server))
    {
      if (OB_SUCCESS != (err = send_request(rs, OB_GET_MS_LIST, _dummy_, ms_list)))
      {
        TBSYS_LOG(ERROR, "send_request(GET_MS_LIST)=>%d", err);
      }
      else if (0 >= ms_list.n_servers_)
      {
        TBSYS_LOG(ERROR, "ms_list.n_servers[%d] <= 0", ms_list.n_servers_);
      }
      else
      {
        server = ms_list.servers_[0];
      }
    }
    else if (0 == strcmp("ups", _server))
    {
      if (OB_SUCCESS != (err = send_request(rs, OB_GET_UPDATE_SERVER_INFO, _dummy_, server)))
      {
        TBSYS_LOG(ERROR, "send_request()=>%d", err);
      }
    }
    return err;
  }

  int scan(const char* rs, const char* table, const char* columns, const char* rowkey, int64_t limit, const char* _server)
  {
    int err = OB_SUCCESS;
    char cbuf[MAX_STR_LEN];
    ObDataBuffer buf(cbuf, sizeof(cbuf));
    ObServer ups;
    int64_t frozen_version = 0;
    ObScanParam scan_param;
    ObServer server;
    ObScanner scanner;
    char* result = NULL;
    if (OB_SUCCESS != (err = send_request(rs, OB_GET_UPDATE_SERVER_INFO, _dummy_, ups)))
    {
      TBSYS_LOG(ERROR, "send_request(get_ups)=>%d", err);
    }
    else if (OB_SUCCESS != (err = send_request(ups, OB_UPS_GET_LAST_FROZEN_VERSION, _dummy_, frozen_version)))
    {
      TBSYS_LOG(ERROR, "send_request(get_last_frozen_version)=>%d", err);
    }
    else if (OB_SUCCESS != (err = choose_server(rs, _server, server)))
    {
      TBSYS_LOG(ERROR, "choose_server(rs=%s, server=%s)=>%d", rs, _server, err);
    }
    else if (OB_SUCCESS != (err = scan_func2(buf, scan_param, frozen_version + 1, table, columns, rowkey, limit)))
    {
      TBSYS_LOG(ERROR, "scan_func2()=>%d", err);
    }
    else if (OB_SUCCESS != (err = send_request(server, OB_SCAN_REQUEST, scan_param, scanner)))
    {
      TBSYS_LOG(ERROR, "send_request(scan)=>%d", err);
      scan_param.dump();
    }
    else if (NULL == (result = buf.get_data() + buf.get_position()))
    {}
    else if (OB_SUCCESS != (err = repr(buf, scanner)))
    {
      TBSYS_LOG(ERROR, "repr(scanner)=>%d", err);
    }
    else
    {
      printf("%s\n", result);
    }
    return err;
  }

  int set(const char* rs, const char* table, const char* column, const char* rowkey, const char* value, const char* _server)
  {
    int err = OB_SUCCESS;
    char cbuf[MAX_STR_LEN];
    ObDataBuffer buf(cbuf, sizeof(cbuf));
    ObSchemaManagerV2 schema_mgr;
    ObServer server;
    ObMutator mutator;
    if (OB_SUCCESS != (err = send_request(rs, OB_FETCH_SCHEMA, _dummy_, schema_mgr)))
    {
      TBSYS_LOG(ERROR, "send_request()=>%d", err);
    }
    else if (OB_SUCCESS != (err = choose_server(rs, _server, server)))
    {
      TBSYS_LOG(ERROR, "choose_server(rs=%s, server=%s)=>%d", rs, _server, err);
    }
    else if (OB_SUCCESS != (err = mutate_func(buf, mutator, schema_mgr, table, rowkey, column, value)))
    {
      TBSYS_LOG(ERROR, "mutate_func()=>%d", err);
    }
    else if (OB_SUCCESS != (err = send_request(server, OB_WRITE, mutator, _dummy_)))
    {
      TBSYS_LOG(ERROR, "send_request(server=%s, write)=>%d", server.to_cstring(), err);
    }
    else
    {
      printf("%s mutate OK!\n", server.to_cstring());
    }
    return err;
  }

  int send_mutator(const char* rs, const char* log_file, const char* _ups)
  {
    int err = OB_SUCCESS;
    int send_err = OB_SUCCESS;
    int64_t pos = 0;
    ObSchemaManagerV2 schema_mgr;
    ObServer ups;
    ObLogEntry entry;
    ObUpsMutator mutator;
    ObMutator final_mutator;
    const char* buf = NULL;
    int64_t len = 0;
    const bool use_name = true;
    const bool show_mutator = false;
    const bool keep_going_on_err = true;
    TBSYS_LOG(DEBUG, "send_mutator(ups=%s, src=%s)", _ups, log_file);
    if (NULL == _ups || NULL == log_file)
    {
      err = OB_INVALID_ARGUMENT;
    }
    else if (OB_SUCCESS != (err = send_request(rs, OB_FETCH_SCHEMA, _dummy_, schema_mgr)))
    {
      TBSYS_LOG(ERROR, "send_request(OB_FETCH_SCHEMA)=>%d", err);
    }
    else if (OB_SUCCESS != (err = choose_server(rs, _ups, ups)))
    {
      TBSYS_LOG(ERROR, "choose_server(rs=%s, _ups=%s)=>%d", rs, _ups, err);
    }
    else if (OB_SUCCESS != (err = get_file_len(log_file, len)))
    {
      TBSYS_LOG(ERROR, "get_file_len(%s)=>%d", log_file, err);
    }
    else if (OB_SUCCESS != (err = file_map_read(log_file, len, buf)))
    {
      TBSYS_LOG(ERROR, "file_map_read(%s)=>%d", log_file, err);
    }
    while(OB_SUCCESS == err && pos < len)
    {
      if (OB_SUCCESS != (err = entry.deserialize(buf, len, pos)))
      {
        TBSYS_LOG(ERROR, "log_entry.deserialize()=>%d", err);
      }
      else
      {
        int64_t tmp_pos = 0;
        if (OB_LOG_UPS_MUTATOR != entry.cmd_)
        {
          TBSYS_LOG(DEBUG, "ignore non mutator[seq=%ld, cmd=%d]", entry.seq_, entry.cmd_);
        }
        else if (OB_SUCCESS != (err = mutator.deserialize(buf + pos, entry.get_log_data_len(), tmp_pos)))
        {
          TBSYS_LOG(ERROR, "mutator.deserialize(seq=%ld)=>%d", (int64_t)entry.seq_, err);
        }
        else if (!mutator.is_normal_mutator())
        {
          TBSYS_LOG(DEBUG, "ignore special mutator[seq=%ld, cmd=%d]", entry.seq_, entry.cmd_);
        }
        else if (use_name && OB_SUCCESS != (err = ob_mutator_resolve_name(schema_mgr, mutator.get_mutator())))
        {
          TBSYS_LOG(ERROR, "mutator_resolve_name()=>%d", err);
        }
        else if (show_mutator && OB_SUCCESS != (err = dump_ob_mutator(mutator.get_mutator())))
        {
          TBSYS_LOG(ERROR, "dump_mutator()=>%d", err);
        }
        else if (OB_SUCCESS != (err = final_mutator.reset()))
        {
          TBSYS_LOG(ERROR, "final_mutator.reset()=>%d", err);
        }
        else if (OB_SUCCESS != (err = mutator_add(final_mutator, mutator.get_mutator(), OB_MAX_PACKET_LENGTH)))
        {
          TBSYS_LOG(ERROR, "mutator_add()=>%d", err);
        }
        else if (OB_SUCCESS != (send_err = send_request(ups, OB_WRITE, final_mutator, _dummy_)))
        {
          TBSYS_LOG(ERROR, "FAIL TO SEND MUTATOR: server=%s, seq=%ld, err=%d", ups.to_cstring(), entry.seq_, send_err);
        }
        else
        {
          TBSYS_LOG(DEBUG, "SUCCESS TO SEND MUTATOR: server=%s, seq=%ld", ups.to_cstring(), entry.seq_);
        }
        if (!keep_going_on_err && OB_SUCCESS != err)
        {
          err = send_err;
        }
        if (OB_SUCCESS == err)
        {
          pos += entry.get_log_data_len();
        }
      }
    }
    if (OB_SUCCESS == err && pos != len)
    {
      err = OB_ERR_UNEXPECTED;
      TBSYS_LOG(ERROR, "pos[%ld] != len[%ld]", pos, len);
    }
    return err;
  }

  int get_clog_status(const char* ups)
  {
    int err = OB_SUCCESS;
    char buf[MAX_STR_LEN];
    ObUpsCLogStatus stat;
    if (NULL == ups)
    {
      err = OB_INVALID_ARGUMENT;
    }
    else if (OB_SUCCESS != (err = send_request(ups, OB_GET_CLOG_STATUS, _dummy_, stat)))
    {
      TBSYS_LOG(ERROR, "send_request(ups=%s, OB_GET_CLOG_STATUS)=>%d", ups, err);
    }
    else if (OB_SUCCESS != (err = stat.to_str(buf, sizeof(buf))))
    {
      TBSYS_LOG(ERROR, "stat.to_str()=>%d", err);
    }
    else
    {
      printf("%s\n", buf);
    }
    return err;
  }
  int get_log_sync_delay_stat(const char* ups)
  {
    int err = OB_SUCCESS;
    ObLogSyncDelayStat delay_stat;
    if (NULL == ups)
    {
      err = OB_INVALID_ARGUMENT;
    }
    else if (OB_SUCCESS != (err = send_request(ups, OB_GET_LOG_SYNC_DELAY_STAT, _dummy_, delay_stat)))
    {
      TBSYS_LOG(ERROR, "send_request(ups=%s, OB_GET_LOG_SYNC_DELAY_STAT)=>%d", ups, err);
    }
    else
    {
      time_t tm = delay_stat.get_last_replay_time_us()/1000000;
      char* str_time = ctime(&tm);
      fprintf(stdout, "log_sync_delay: last_log_id=%ld, total_count=%ld, total_delay=%ldus, max_delay=%ldus, last_replay_time=%ldus [%s]\n",
              delay_stat.get_last_log_id(), delay_stat.get_mutator_count(), delay_stat.get_total_delay_us(),
              delay_stat.get_max_delay_us(), delay_stat.get_last_replay_time_us(),
              str_time);
    }
    return err;
  }
  bool is_init_;
};

#define report_error(err, ...) if (OB_SUCCESS != err)TBSYS_LOG(ERROR, __VA_ARGS__);
#include "cmd_args_parser.h"
int main(int argc, char *argv[])
{
  int err = 0;
  RPC rpc;
  TBSYS_LOGGER.setLogLevel(getenv("log_level")?:"WARN");
  if (getenv("log_file"))
    TBSYS_LOGGER.setFileName(getenv("log_file"));
  if (OB_SUCCESS != (err = ob_init_memory_pool()))
  {
    TBSYS_LOG(ERROR, "ob_init_memory_pool()=>%d", err);
  }
  else if (OB_SUCCESS != (err = rpc.initialize()))
  {
    TBSYS_LOG(ERROR, "rpc.initialize()=>%d", err);
  }
  else if (OB_NEED_RETRY != (err = CmdCall(argc, argv, rpc.desc, StrArg(rs), StrArg(table, "*")):OB_NEED_RETRY))
  {
    report_error(err, "desc()=>%d", err);
  }
  else if (OB_NEED_RETRY != (err = CmdCall(argc, argv, rpc.get_obi_role, StrArg(rs)):OB_NEED_RETRY))
  {
    report_error(err, "get_obi_role()=>%d", err);
  }
  else if (OB_NEED_RETRY != (err = CmdCall(argc, argv, rpc.get_master_ups, StrArg(rs)):OB_NEED_RETRY))
  {
    report_error(err, "get_master_ups()=>%d", err);
  }
  else if (OB_NEED_RETRY != (err = CmdCall(argc, argv, rpc.get_client_cfg, StrArg(rs)):OB_NEED_RETRY))
  {
    report_error(err, "get_client_cfg()=>%d", err);
  }
  else if (OB_NEED_RETRY != (err = CmdCall(argc, argv, rpc.get_replayed_cursor, StrArg(ups)):OB_NEED_RETRY))
  {
    report_error(err, "get_replayed_cursor()=>%d", err);
  }
  else if (OB_NEED_RETRY != (err = CmdCall(argc, argv, rpc.get_max_log_seq_replayable, StrArg(ups)):OB_NEED_RETRY))
  {
    report_error(err, "get_max_log_seq()=>%d", err);
  }
  else if (OB_NEED_RETRY != (err = CmdCall(argc, argv, rpc.get_last_frozen_version, StrArg(ups)):OB_NEED_RETRY))
  {
    report_error(err, "get_last_frozen_version()=>%d", err);
  }
  else if (OB_NEED_RETRY != (err = CmdCall(argc, argv, rpc.get_ms, StrArg(rs)):OB_NEED_RETRY))
  {
    report_error(err, "get_ms()=>%d", err);
  }
  else if (OB_NEED_RETRY != (err = CmdCall(argc, argv, rpc.scan, StrArg(rs), StrArg(table),
                                           StrArg(columns, "*"), StrArg(rowkey, "[min,max]"), IntArg(limit, "10"),  StrArg(server, "ms")):OB_NEED_RETRY))
  {
    report_error(err, "scan()=>%d", err);
  }
  else if (OB_NEED_RETRY != (err = CmdCall(argc, argv, rpc.set, StrArg(rs), StrArg(table),
                                           StrArg(column), StrArg(rowkey), StrArg(value), StrArg(server, "ups")):OB_NEED_RETRY))
  {
    report_error(err, "set()=>%d", err);
  }
  else if (OB_NEED_RETRY != (err = CmdCall(argc, argv, rpc.send_mutator, StrArg(rs), StrArg(log_file), StrArg(server, "ups")):OB_NEED_RETRY))
  {
    report_error(err, "send_mutator()=>%d", err);
  }
  else if (OB_NEED_RETRY != (err = CmdCall(argc, argv, rpc.get_log_sync_delay_stat, StrArg(ups)):OB_NEED_RETRY))
  {
    report_error(err, "get_log_sync_delay_stat()=>%d", err);
  }
  else if (OB_NEED_RETRY != (err = CmdCall(argc, argv, rpc.get_clog_status, StrArg(ups)):OB_NEED_RETRY))
  {
    report_error(err, "get_clog_status()=>%d", err);
  }
  else
  {
    fprintf(stderr, _usages, argv[0]);
    //__cmd_args_parser.dump(argc, argv);
  }
  exit(err);
  return err;
}
