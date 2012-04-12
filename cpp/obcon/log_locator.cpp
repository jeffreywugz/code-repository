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
#include "updateserver/ob_on_disk_log_locator.h"
#include "common/ob_direct_log_reader.h"
#include "stdlib.h"

using namespace oceanbase::common;
using namespace oceanbase::updateserver;

const char* _usages = "Usages:\n"
  "\t# You can set env var 'log_level' to 'DEBUG'/'WARN'...\n"
  "\t%1$s locate log_dir/end_file_id log_id\n"
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
    TBSYS_LOG(ERROR, "find_log(log_dir=%s, log_id=%d): INVALID ARGUMENT.", log_dir, log_id);
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
      fprintf(stdout, "SEQ: %lu\tPayload Length: %ld\tTYPE: %d\n", log_seq, data_len, cmd);
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
