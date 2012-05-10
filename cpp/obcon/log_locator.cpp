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

#include "tbsys.h"
#include "common/ob_malloc.h"
#include "updateserver/ob_ups_mutator.h"
#include "updateserver/ob_on_disk_log_locator.h"
#include "common/ob_direct_log_reader.h"
#include "stdlib.h"

using namespace oceanbase::common;
using namespace oceanbase::updateserver;

const char* _usages = "Usages:\n"
  "\t# You can set env var 'log_level' to 'DEBUG'/'WARN'...\n"
  "\t%1$s locate log_dir log_id\n"
  "\t%1$s dump log_dir/log_file_id\n";

int dump_log_location(ObOnDiskLogLocator& log_locator, int64_t log_id)
{
  int err = OB_SUCCESS;
  ObLogLocation location;
  if (OB_SUCCESS != (err = log_locator.get_location(log_id, location)))
  {
    TBSYS_LOG(ERROR, "log_locator.get_location(%ld)=>%d", log_id, err);
  }
  else
  {
    printf("%ld -> [file_id=%ld, offset=%ld, log_id=%ld]\n", log_id, location.file_id_, location.offset_, location.log_id_);
  }
  return err;
}

int locate(const char* log_dir, int64_t log_id)
{
  int err = OB_SUCCESS;
  ObOnDiskLogLocator log_locator;
  ObLogLocation location;
  if (NULL == log_dir || 0 >= log_id)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "find_log(log_dir=%s, log_id=%ld): INVALID ARGUMENT.", log_dir, log_id);
  }
  else if (OB_SUCCESS != (err = log_locator.init(log_dir)))
  {
    TBSYS_LOG(ERROR, "log_locator.init(%s)=>%d", log_dir, err);
  }
  else if (OB_SUCCESS != (err = dump_log_location(log_locator, log_id)))
  {
    TBSYS_LOG(ERROR, "dump_log_location(%ld)=>%d", log_id, err);
  }
  return err;
}

static const char* LogCmdStr[1024];
int init_log_cmd_str()
{
  memset(LogCmdStr, 0x00, sizeof(LogCmdStr));
  LogCmdStr[OB_LOG_SWITCH_LOG]            = "SWITCH_LOG";
  LogCmdStr[OB_LOG_CHECKPOINT]            = "CHECKPOINT";
  LogCmdStr[OB_LOG_NOP]                   = "NOP";
  LogCmdStr[OB_LOG_UPS_MUTATOR]           = "UPS_MUTATOR";
  LogCmdStr[OB_UPS_SWITCH_SCHEMA]         = "UPS_SWITCH_SCHEMA";
  LogCmdStr[OB_RT_SCHEMA_SYNC]            = "OB_RT_SCHEMA_SYNC";
  LogCmdStr[OB_RT_CS_REGIST]              = "OB_RT_CS_REGIST";
  LogCmdStr[OB_RT_MS_REGIST]              = "OB_RT_MS_REGIST";
  LogCmdStr[OB_RT_SERVER_DOWN]            = "OB_RT_SERVER_DOWN";
  LogCmdStr[OB_RT_CS_LOAD_REPORT]         = "OB_RT_CS_LOAD_REPORT";
  LogCmdStr[OB_RT_CS_MIGRATE_DONE]        = "OB_RT_CS_MIGRATE_DONE";
  LogCmdStr[OB_RT_CS_START_SWITCH_ROOT_TABLE]
    = "OB_RT_CS_START_SWITCH_ROOT_TABLE";
  LogCmdStr[OB_RT_START_REPORT]           = "OB_RT_START_REPORT";
  LogCmdStr[OB_RT_REPORT_TABLETS]         = "OB_RT_REPORT_TABLETS";
  LogCmdStr[OB_RT_ADD_NEW_TABLET]         = "OB_RT_ADD_NEW_TABLET";
  LogCmdStr[OB_RT_CREATE_TABLE_DONE]      = "OB_RT_CREATE_TABLE_DONE";
  LogCmdStr[OB_RT_BEGIN_BALANCE]          = "OB_RT_BEGIN_BALANCE";
  LogCmdStr[OB_RT_BALANCE_DONE]           = "OB_RT_BALANCE_DONE";
  LogCmdStr[OB_RT_US_MEM_FRZEEING]        = "OB_RT_US_MEM_FRZEEING";
  LogCmdStr[OB_RT_US_MEM_FROZEN]          = "OB_RT_US_MEM_FROZEN";
  LogCmdStr[OB_RT_CS_START_MERGEING]      = "OB_RT_CS_START_MERGEING";
  LogCmdStr[OB_RT_CS_MERGE_OVER]          = "OB_RT_CS_MERGE_OVER";
  LogCmdStr[OB_RT_CS_UNLOAD_DONE]         = "OB_RT_CS_UNLOAD_DONE";
  LogCmdStr[OB_RT_US_UNLOAD_DONE]         = "OB_RT_US_UNLOAD_DONE";
  LogCmdStr[OB_RT_DROP_CURRENT_BUILD]     = "OB_RT_DROP_CURRENT_BUILD";
  LogCmdStr[OB_RT_DROP_LAST_CS_DURING_MERGE]
    = "OB_RT_DROP_LAST_CS_DURING_MERGE";
  LogCmdStr[OB_RT_SYNC_FROZEN_VERSION]    = "OB_RT_SYNC_FROZEN_VERSION";
  LogCmdStr[OB_RT_SET_UPS_LIST]           = "OB_RT_SET_UPS_LIST";
  LogCmdStr[OB_RT_SET_CLIENT_CONFIG]      = "OB_RT_SET_CLIENT_CONFIG";
  return 0;
}

const char* get_log_cmd_repr(const LogCommand cmd)
{
  const char* cmd_repr = NULL;
  if (cmd < 0 || cmd >= (int)ARRAYSIZEOF(LogCmdStr))
  {}
  else
  {
    cmd_repr = LogCmdStr[cmd];
  }
  if (NULL == cmd_repr)
  {
    cmd_repr = "unknown";
  }
  return cmd_repr;
}

const char* format_time(int64_t time_us)
{
  static char time_str[1024];
  const char* format = "%Y-%m-%d %H:%M:%S";
  struct tm time_struct;
  int64_t time_s = time_us / 1000000;
  if(NULL != localtime_r(&time_s, &time_struct))
  {
    strftime(time_str, sizeof(time_str), format, &time_struct);
  }
  time_str[sizeof(time_str)-1] = 0;
  return time_str;
}

int dump_mutator(ObUpsMutator& mutator)
{
  int err = OB_SUCCESS;
  printf("MutationTime: %s checksum %ld:%ld\n",
         format_time(mutator.get_mutate_timestamp()),
         mutator.get_memtable_checksum_before_mutate(),
         mutator.get_memtable_checksum_after_mutate());
  return err;
}

int dump(const char* log_file)
{
  int err = OB_SUCCESS;
  ObDirectLogReader reader;
  LogCommand cmd = OB_LOG_UNKNOWN;
  uint64_t log_seq = 0;
  char* log_data = NULL;
  int64_t data_len = 0;
  char* p = NULL;
  const char* log_dir = NULL;
  const char* log_name = NULL;
  ObUpsMutator mutator;
  if (NULL == log_file)
  {
    err = OB_INVALID_ARGUMENT;
  }
  else if (NULL == (p = strrchr(log_file, '/')))
  {
    log_dir = ".";
    log_name = log_file;
  }
  else
  {
    log_dir = log_file;
    *p = '\0';
    log_name = p + 1;
  }

  if (OB_SUCCESS != err)
  {}
  else if (OB_SUCCESS != (err = reader.init(log_dir)))
  {
    TBSYS_LOG(ERROR, "reader.init(log_dir=%s)=>%d", log_dir, err);
  }
  else if (OB_SUCCESS != (err = reader.open(atoi(log_name))))
  {
    TBSYS_LOG(ERROR, "reader.open(log_name=%s)=>%d", log_name, err);
  }

  while(OB_SUCCESS == err)
  {
    if (OB_SUCCESS != (err = reader.read_log(cmd, log_seq, log_data, data_len)) && OB_READ_NOTHING != err)
    {
      TBSYS_LOG(ERROR, "read_log()=>%d", err);
    }
    else
    {
      int64_t pos = 0;
      fprintf(stdout, "%lu|%ld\t|%ld\t%s[%d]\n", log_seq, reader.get_cur_offset(), data_len, get_log_cmd_repr(cmd), cmd);
      if (OB_LOG_UPS_MUTATOR != cmd)
      {}
      else if (OB_SUCCESS != (err = mutator.deserialize(log_data, data_len, pos)))
      {
        TBSYS_LOG(ERROR, "mutator.deserialize(seq=%ld)=>%d", log_seq, err);
      }
      else if (OB_SUCCESS != (err = dump_mutator(mutator)))
      {
        TBSYS_LOG(ERROR, "dump_mutator()=>%d", err);
      }
    }
  }
  if (OB_READ_NOTHING == err)
  {
    err = OB_SUCCESS;
  }
  return err;
}

#include "cmd_args_parser.h"
int main(int argc, char *argv[])
{
  int err = 0;
  TBSYS_LOGGER.setLogLevel(getenv("log_level")?:"INFO");
  init_log_cmd_str();
  if (OB_SUCCESS != (err = ob_init_memory_pool()))
  {
    TBSYS_LOG(ERROR, "ob_init_memory_pool()=>%d", err);
  }
  else if (OB_NEED_RETRY != (err = CmdCall(argc, argv, dump, StrArg(log_file)):OB_NEED_RETRY))
  {
    if (OB_SUCCESS != err)
    {
      TBSYS_LOG(ERROR, "dump()=>%d", err);
    }
  }
  else if (OB_NEED_RETRY != (err = CmdCall(argc, argv, locate, StrArg(log_dir), IntArg(log_id)):OB_NEED_RETRY))
  {
    if (OB_SUCCESS != err)
    {
      TBSYS_LOG(ERROR, "locate()=>%d", err);
    }
  }
  else
  {
    fprintf(stderr, _usages, argv[0]);
  }
  return err;
}
