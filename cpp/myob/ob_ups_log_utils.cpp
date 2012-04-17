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

#include "ob_ups_log_utils.h"
#include "ob_sstable_mgr.h"
//#include "common/ob_file_client.h"
#include "common/ob_log_reader.h"
#include "common/ob_direct_log_reader.h"
#include "common/ob_log_dir_scanner.h"
#include "common/file_utils.h"
#include "common/file_directory_utils.h"
#include "ob_ups_table_mgr.h"

using namespace oceanbase::common;
namespace oceanbase
{
  namespace updateserver
  {
    int load_replay_point_func(const char* log_dir, int64_t& replay_point)
    {
      int err = 0;
      int64_t len = 0;
      int open_ret = 0;
      char rplpt_fn[OB_MAX_FILE_NAME_LENGTH];
      char rplpt_str[ObUpsLogMgr::UINT64_MAX_LEN];
      int rplpt_str_len = 0;
      FileUtils rplpt_file;

      if (NULL == log_dir)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "Arguments are invalid[log_dir=%p]", log_dir);
      }
      else if ((len = snprintf(rplpt_fn, sizeof(rplpt_fn), "%s/%s", log_dir, ObUpsLogMgr::UPS_LOG_REPLAY_POINT_FILE) < 0)
               && len >= (int64_t)sizeof(rplpt_fn))
      {
        err = OB_ERROR;
        TBSYS_LOG(ERROR, "generate_replay_point_fn()=>%d", err);
      }
      else if (!FileDirectoryUtils::exists(rplpt_fn))
      {
        err = OB_FILE_NOT_EXIST;
        TBSYS_LOG(WARN, "replay point file[\"%s\"] does not exist", rplpt_fn);
      }
      else if (0 > (open_ret = rplpt_file.open(rplpt_fn, O_RDONLY)))
      {
        err = OB_IO_ERROR;
        TBSYS_LOG(WARN, "open file[\"%s\"] error[%s]", rplpt_fn, strerror(errno));
      }
      else if (0 > (rplpt_str_len = rplpt_file.read(rplpt_str, sizeof(rplpt_str)))
               || rplpt_str_len >= (int64_t)sizeof(rplpt_str))
      {
        err = rplpt_str_len < 0 ? OB_IO_ERROR: OB_BUF_NOT_ENOUGH;
        TBSYS_LOG(ERROR, "read error[%s] or file contain invalid data[len=%d]", strerror(errno), rplpt_str_len);
      }
      else
      {
        rplpt_str[rplpt_str_len] = '\0';
        const int STRTOUL_BASE = 10;
        char* endptr;
        replay_point = strtoul(rplpt_str, &endptr, STRTOUL_BASE);
        if ('\0' != *endptr)
        {
          err = OB_INVALID_DATA;
          TBSYS_LOG(ERROR, "non-digit exist in replay point file[rplpt_str=%.*s]", rplpt_str_len, rplpt_str);
        }
        else if (ERANGE == errno)
        {
          err = OB_INVALID_DATA;
          TBSYS_LOG(ERROR, "replay point contained in replay point file is out of range");
        }
      }
      if (0 > open_ret)
      {
        rplpt_file.close();
      }
      return err;
    }

    int write_replay_point_func(const char* log_dir, const int64_t replay_point)
    {
      int err = OB_SUCCESS;
      int len = 0;
      int open_ret = 0;
      FileUtils rplpt_file;
      char rplpt_fn[OB_MAX_FILE_NAME_LENGTH];
      char rplpt_str[ObUpsLogMgr::UINT64_MAX_LEN];
      int rplpt_str_len = 0;

      if (NULL == log_dir)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "Arguments are invalid[log_dir=%p]", log_dir);
      }
      else if ((len = snprintf(rplpt_fn, sizeof(rplpt_fn), "%s/%s", log_dir, ObUpsLogMgr::UPS_LOG_REPLAY_POINT_FILE) < 0)
               && len >= (int64_t)sizeof(rplpt_fn))
      {
        err = OB_BUF_NOT_ENOUGH;
        TBSYS_LOG(ERROR, "generate_replay_point_fn()=>%d", err);
      }
      else if (0 > (open_ret = rplpt_file.open(rplpt_fn, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH)))
      {
        err = OB_FILE_NOT_EXIST;
        TBSYS_LOG(ERROR, "open file[\"%s\"] error[%s]", rplpt_fn, strerror(errno));
      }
      else if ((rplpt_str_len = snprintf(rplpt_str, sizeof(rplpt_str), "%lu", replay_point)) < 0
               || rplpt_str_len >= (int64_t)sizeof(rplpt_str))
      {
        err = OB_BUF_NOT_ENOUGH;
        TBSYS_LOG(ERROR, "snprintf rplpt_str error[%s][replay_point=%lu]", strerror(errno), replay_point);
      }
      else if (0 > (rplpt_str_len = rplpt_file.write(rplpt_str, rplpt_str_len)))
      {
        err = OB_ERR_SYS;
        TBSYS_LOG(ERROR, "write error[%s][rplpt_str=%p rplpt_str_len=%d]", strerror(errno), rplpt_str, rplpt_str_len);
      }
      if (0 > open_ret)
      {
        rplpt_file.close();
      }
      return err;
    }

    int scan_log_dir_func(const char* log_dir, int64_t& replay_point, int64_t& min_log_file_id, int64_t& max_log_file_id)
    {
      int err = OB_SUCCESS;
      ObLogDirScanner scanner;
      min_log_file_id = 0;
      max_log_file_id = 0;
      replay_point = 0;
      if (NULL == log_dir)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "log_dir == NULL");
      }
      else if (OB_SUCCESS != (err = scanner.init(log_dir))
               && OB_DISCONTINUOUS_LOG != err)
      {
        TBSYS_LOG(ERROR, "scanner.init(log_dir=%s)=>%d", log_dir, err);
      }
      else if (OB_SUCCESS != (err = scanner.get_min_log_file_id((uint64_t&)min_log_file_id)) && OB_ENTRY_NOT_EXIST != err)
      {
        TBSYS_LOG(ERROR, "scanner.get_min_log_file_id()=>%d", err);
      }
      else if (OB_SUCCESS != (err = scanner.get_max_log_file_id((uint64_t&)max_log_file_id)) && OB_ENTRY_NOT_EXIST != err)
      {
        TBSYS_LOG(ERROR, "get_max_log_file_id error[ret=%d]", err);
      }
      else if (OB_SUCCESS != (err = load_replay_point_func(log_dir, replay_point)) && OB_FILE_NOT_EXIST != err)
      {
        TBSYS_LOG(ERROR, "load_replay_point(log_dir=%s)=>%d", log_dir, err);
      }
      else
      {
        if (0 >= min_log_file_id)
        {
          min_log_file_id = 1;
        }
        if (0 >= replay_point)
        {
          replay_point = min_log_file_id;
        }
        if (min_log_file_id > replay_point || max_log_file_id >0 && replay_point > max_log_file_id)
        {
          err = OB_DISCONTINUOUS_LOG;
          TBSYS_LOG(WARN, "min_log_file_id=%ld, max_log_file_id=%ld, replay_point=%ld", min_log_file_id, max_log_file_id, replay_point);
        }
      }
      if (OB_FILE_NOT_EXIST == err)
      {
        err = OB_SUCCESS;
      }
      return err;
    }

    int get_replay_point_func(const char* log_dir, int64_t& replay_point)
    {
      int64_t min_log_id = -1;
      int64_t max_log_id = -1;
      return scan_log_dir_func(log_dir, replay_point, min_log_id, max_log_id);
    }
    
    int get_local_max_log_cursor_func(const char* log_dir, const int64_t log_file_id_by_sst, ObLogCursor& log_cursor)
    {
      int err = OB_SUCCESS;
      int64_t max_log_file_id = 0;
      ObLogDirScanner scanner;
      if (NULL == log_dir)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "log_dir == NULL");
      }
      else if (OB_SUCCESS != (err = scanner.init(log_dir))
               && OB_DISCONTINUOUS_LOG != err)
      {
        TBSYS_LOG(ERROR, "scanner.init(log_dir=%s)=>%d", log_dir, err);
      }
      else if (OB_SUCCESS != (err = scanner.get_max_log_file_id((uint64_t&)max_log_file_id)) && OB_ENTRY_NOT_EXIST != err)
      {
        TBSYS_LOG(ERROR, "get_max_log_file_id error[ret=%d]", err);
      }
      if (OB_INVALID_ID != log_file_id_by_sst && max_log_file_id < log_file_id_by_sst)
      {
        max_log_file_id = log_file_id_by_sst;
      }
      if (max_log_file_id > 0 && 
      {
      }
      return err;
    }
    // int seek_log_by_id(int64_t seq_id, int64_t& n_skiped, const char* log_data, int64_t data_len, int64_t& pos)
    // {
    //   int err = OB_SUCCESS;
    //   ObLogEntry log_entry;
    //   int64_t id = 0;
    //   n_skiped = 0;
    //   if (NULL == log_data || data_len <= 0 || seq_id < 0)
    //   {
    //     err = OB_INVALID_ARGUMENT;
    //     TBSYS_LOG(ERROR, "invalid argument, log_data=%p, data_len=%ld", log_data, data_len);
    //   }
    //   while (OB_SUCCESS == err && pos < data_len && id < seq_id)
    //   {
    //     if (OB_SUCCESS != (err = log_entry.deserialize(log_data, data_len, pos)))
    //     {
    //       TBSYS_LOG(ERROR, "log_entry.deserialize(log_data=%p, data_len=%ld, pos=%ld)=>%d", log_data, data_len, pos, err);
    //     }
    //     else if (OB_SUCCESS != (err = log_entry.check_data_integrity(log_data + pos)))
    //     {
    //       TBSYS_LOG(ERROR, "log_entry.check_data_integrity()=>%d", err);
    //     }
    //     else
    //     {
    //       n_skiped++;
    //       pos += log_entry.get_log_data_len();
    //       id = log_entry.seq_;
    //     }
    //   }
    //   if (OB_SUCCESS != err)
    //   {}
    //   else if (id  > seq_id + 1)
    //   {
    //     err = OB_ERR_UNEXPECTED;
    //     TBSYS_LOG(ERROR, "crop_log_task(id=%ld, id_to_seek=%ld)=>%d", id, seq_id, err);
    //   }
    //   else if (id == seq_id + 1)
    //   {
    //     pos -= log_entry.get_serialize_size() + log_entry.get_log_data_len();
    //     n_skiped--;
    //   }
    //   return err;
    // }

    // int crop_log_task(ObBatchLogEntryTask* tasks, int64_t seq_id)
    // {
    //   int err = OB_SUCCESS;
    //   int64_t pos = 0;
    //   int64_t n_skiped = 0;
    //   TBSYS_LOG(INFO, "crop_log_task(tasks[min_id=%ld, max_id=%ld], seek_id=%ld, log_data_len=%ld)", tasks->min_id_, tasks->max_id_, seq_id, tasks->log_data_len_);
    //   if (NULL == tasks || !tasks->is_valid() ||
    //       seq_id < 0 || seq_id > 0 && seq_id + 1 < tasks->min_id_ || seq_id > tasks->max_id_)
    //   {
    //     err = OB_INVALID_ARGUMENT;
    //     TBSYS_LOG(ERROR, "crop_log_task(seq_id=%ld, min_id=%ld, max_id=%ld)=>%d", seq_id, tasks->min_id_, tasks->max_id_, err);
    //   }
    //   else if (0 == seq_id)
    //   {
    //     seq_id = tasks->min_id_-1;
    //   }

    //   if (OB_SUCCESS != err)
    //   {}
    //   else if (OB_SUCCESS != (err = seek_log_by_id(seq_id, n_skiped, tasks->log_data_, tasks->log_data_len_, pos)))
    //   {
    //     TBSYS_LOG(ERROR, "seek_log_by_id(seq_id=%ld, log_data=%p, data_len=%ld)=>%d", seq_id, tasks->log_data_, tasks->log_data_len_, err);
    //   }
    //   else
    //   {
    //     //TBSYS_LOG(INFO, "seek_by_id(pos=%ld, n_skiped=%ld, seq_id=%ld)", pos, n_skiped, seq_id);
    //     tasks->log_data_  += pos;
    //     tasks->log_data_len_ -= pos;
    //     tasks->n_log_tasks_ -= n_skiped;
    //     tasks->log_tasks_ += n_skiped;
    //     tasks->min_id_ = seq_id;
    //   }
    //   return err;
    // }

    int parse_log_buffer(const char* log_data, const int64_t len, const int64_t limit, int64_t& end_pos,
                         const ObLogCursor& start_cursor, ObLogCursor& end_cursor)
    {
      int err = OB_SUCCESS;
      int64_t pos = 0;
      int64_t file_id = 0;
      ObLogEntry log_entry;
      end_cursor = start_cursor;
      if (NULL == log_data || len <= 0 || limit <= 0 || len < limit || !start_cursor.is_valid())
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "invalid argument, log_data=%p, limit=%ld, start_cursor=%s",
                  log_data, limit, start_cursor.to_str());
      }

      while (OB_SUCCESS == err && pos + log_entry.get_serialize_size() < limit)
      {
        int64_t old_pos = pos;
        if (OB_SUCCESS != (err = log_entry.deserialize(log_data, len, pos)))
        {
          TBSYS_LOG(ERROR, "log_entry.deserialize(log_data=%p, len=%ld, pos=%ld)=>%d", log_data, len, pos, err);
        }
        else if (old_pos + log_entry.get_serialize_size() + log_entry.get_log_data_len() > limit)
        {
          pos = old_pos;
          break;
        }
        else if (OB_SUCCESS != (err = log_entry.check_data_integrity(log_data + pos)))
        {
          TBSYS_LOG(ERROR, "log_entry.check_data_integrity()=>%d", err);
        }

        if (OB_SUCCESS != err)
        {}
        else if (OB_LOG_SWITCH_LOG == log_entry.cmd_
                 && !(OB_SUCCESS == (err = serialization::decode_i64(log_data, len, pos, (int64_t*)&file_id)
                                     && start_cursor.log_id_ == file_id)))
        {
          TBSYS_LOG(ERROR, "decode switch_log failed(log_data=%p, limit=%ld, pos=%ld)=>%d", log_data, limit, pos, err);
        }
        else
        {
          pos = old_pos + log_entry.get_serialize_size() + log_entry.get_log_data_len();
          if (OB_SUCCESS != (err = end_cursor.advance(log_entry)))
          {
            TBSYS_LOG(ERROR, "end_cursor[%ld].advance(%ld)=>%d", end_cursor.log_id_, log_entry.seq_, err);
          }
        }
      }
      // if (OB_SUCCESS == err && limit == len && pos != limit)
      // {
      //   err = OB_ERR_UNEXPECTED;
      //   TBSYS_LOG(ERROR, "pos[%ld] != limit[%ld]", pos, limit);
      // }

      if (OB_SUCCESS != err && OB_INVALID_ARGUMENT != err)
      {
        TBSYS_LOG(ERROR, "parse log buf error:");
        hex_dump(log_data, limit, true, TBSYS_LOG_LEVEL_WARN);
      }
      else
      {
        end_pos = pos;
      }
      return err;
    }

    int logcpy(char* dest, const char* src, const int64_t len, const int64_t limit, int64_t bytes_cpy,
              const ObLogCursor& start_cursor, ObLogCursor& end_cursor)
    {
      int err = OB_SUCCESS;
      if (NULL == dest || NULL == src || 0 >= len || 0 >= limit || len < limit || !start_cursor.is_valid())
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "logcpy(dest=%p, src=%p, len=%ld, limit=%ld, start_cursor=%s)=>%d",
                  dest, src, len, limit, start_cursor.to_str(), err);
      }
      else if (OB_SUCCESS != (err = parse_log_buffer(src, len, limit, bytes_cpy, start_cursor, end_cursor)))
      {
        TBSYS_LOG(ERROR, "parse_log_buffer(src=%p, len=%ld, limit=%ld, start_cursor=%s)=>%d", src, len, limit, start_cursor.to_str(), err);
      }
      else
      {
        memcpy(dest, src, bytes_cpy);
      }
      return err;
    }

    int join_path(char* result_path, const int64_t len_limit, const char* base, const char* path)
    {
      int err = OB_SUCCESS;
      int64_t len = 0;
      if (NULL == result_path || 0 >= len_limit || NULL == path)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "join_path(result_path=%s, limit=%ld, base=%s, path=%s)=>%d",
                  result_path, len_limit, base, path, err);
      }
      else if ('/' == path[0] || NULL == base)
      {
        if (0 >= (len = snprintf(result_path, len_limit, "%s", path)) || len > len_limit)
        {
          err = OB_BUF_NOT_ENOUGH;
          TBSYS_LOG(ERROR, "snprintf(path=%s)=>%d", path, len);
        }
      }
      else
      {
        if(0 >= (len = snprintf(result_path, len_limit, "%s/%s", base,  path)) || len > len_limit)
        {
          err = OB_BUF_NOT_ENOUGH;
          TBSYS_LOG(ERROR, "snprintf(base=%s, path=%s)=>%d", base, path, err);
        }
      }
      return err;
    }
    
    int get_absolute_path(char* abs_path, const int64_t len_limit, const char* path)
    {
      int err = OB_SUCCESS;
      char cwd[OB_MAX_FILE_NAME_LENGTH];
      if (NULL == abs_path || 0 >= len_limit || NULL == path)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "get_absolute_path(abs_path=%p, limit=%ld, path=%s)=>%d", abs_path, len_limit, path, err);
      }
      else if (NULL == getcwd(cwd, sizeof(cwd)))
      {
        err = OB_BUF_NOT_ENOUGH;
        TBSYS_LOG(ERROR, "getcwd error[%s]", strerror(errno));
      }
      else if (OB_SUCCESS != (err = join_path(abs_path, len_limit, cwd, path)))
      {
        TBSYS_LOG(ERROR, "join_path(len_limit=%ld, base=%s, path=%s)=>%d", len_limit, cwd, path, err);
      }
      return err;
    }
    // static int fetch_log_file(const volatile bool& stop, IObScpHandler* scp, const ObServer& server, const char* log_dir_arg,
    //                           const int64_t min_log_file_id, const int64_t max_log_file_id, int64_t timeout)
    // {
    //   int err = OB_SUCCESS;
    //   int64_t len = 0;
    //   char log_dir[OB_MAX_FILE_NAME_LENGTH];
    //   if (NULL == log_dir_arg || 0 >= min_log_file_id || 0 >= max_log_file_id || min_log_file_id > max_log_file_id)
    //   {
    //     err = OB_INVALID_ARGUMENT;
    //     TBSYS_LOG(ERROR, "fetch_log_file(log_dir=%s, min_id=%ld, max_id=%ld)=>%d", log_dir_arg, min_log_file_id, max_log_file_id, err);
    //   }
    //   else if (OB_SUCCESS != (err = get_absolute_path(log_dir, sizeof(log_dir), log_dir_arg)))
    //   {
    //     TBSYS_LOG(ERROR, "get_absolute_path(log_dir=%s)=>%d", log_dir_arg, err);
    //   }
    //   for (int64_t i = min_log_file_id; OB_SUCCESS == err && !stop && i < max_log_file_id; i++)
    //   {
    //     char file_name[OB_MAX_FILE_NAME_LENGTH];
    //     if (0 >= (len = snprintf(file_name, sizeof(file_name), "%ld", i)) || (int64_t)sizeof(file_name) < len)
    //     {
    //       err = OB_BUF_NOT_ENOUGH;
    //       TBSYS_LOG(WARN, "snprintf(i=%ld)=>%d", i, err);
    //     }
    //     else if(OB_SUCCESS != (err = scp->get_file(server, log_dir, file_name, log_dir, file_name, timeout)))
    //     {
    //       TBSYS_LOG(WARN, "cp_remote_file(dir=%s, file=%s)=>%d", log_dir, file_name, err);
    //     }
    //   }
    //   return err;
    // }
      
    // static int fetch_sst_file(const volatile bool& stop, IObScpHandler* scp, const ObServer& server,
    //                           StoreMgr& store_mgr, const SSTList& sst_list, const int64_t timeout)
    // {
    //   int err = OB_SUCCESS;
    //   char first_store_path[OB_MAX_FILE_NAME_LENGTH];

    //   ObList<StoreMgr::Handle> store_list;

    //   if (NULL == scp)
    //   {
    //     err = OB_INVALID_ARGUMENT;
    //     TBSYS_LOG(ERROR, "scp == NULL");
    //   }
    //   for (SSTList::iterator ssti = sst_list.begin(); OB_SUCCESS == err && !stop && ssti != sst_list.end(); ++ssti)
    //   {
    //     if (OB_SUCCESS != (err = store_mgr.assign_stores(store_list)))
    //     {
    //       TBSYS_LOG(WARN, "ObStoreMgr assign_stores error, err=%d", err);
    //     }
    //     else if (store_list.empty())
    //     {
    //       err = OB_ERROR;
    //       TBSYS_LOG(WARN, "store list is empty, err=%d", err);
    //     }

    //     const char* first_store_dir = NULL;
    //     int counter = 1;
    //     int len = 0;
    //     for (ObList<StoreMgr::Handle>::iterator stoi  = store_list.begin();
    //          OB_SUCCESS == err && stoi != store_list.end(); ++stoi, ++counter)
    //     {
    //       const char* store_dir = NULL;
    //       if (NULL == (store_dir = store_mgr.get_dir(*stoi)))
    //       {
    //         TBSYS_LOG(WARN, "%d dir in store list is NULL", counter);
    //       }
    //       else if (NULL == first_store_dir)
    //       {
    //         if(OB_SUCCESS != (err = scp->get_file(server, ssti->path.ptr(), ssti->name.ptr(), store_dir, ssti->name.ptr(), timeout)))
    //         {
    //           TBSYS_LOG(INFO, "get sstable faile, path=%s name=%s dst_path=%s", ssti->path.ptr(), ssti->name.ptr(), store_dir);
    //         }
    //         else if (0 >= (len = snprintf(first_store_path, sizeof(first_store_path), "%s", store_dir))
    //                  || len > (int)sizeof(first_store_path))
    //         {
    //           err = OB_BUF_NOT_ENOUGH;
    //           TBSYS_LOG(ERROR, "sprintf(len_limit=%ld, path=%s)=>%d", sizeof(first_store_path), store_dir, err);
    //         }
    //         else
    //         {
    //           first_store_dir = first_store_path;
    //           TBSYS_LOG(INFO, "get sstable succ, path=%s name=%s dst_path=%s", ssti->path.ptr(), ssti->name.ptr(), store_dir);
    //         }
    //       }
    //       else if (OB_SUCCESS != (err = FSU::cp_safe(first_store_dir, ssti->name.ptr(), store_dir, ssti->name.ptr())))
    //       {
    //         TBSYS_LOG(ERROR, "cp_safe(src_dir=%s, src_file=%s, dst_dir=%s, dst_file=%s)=>%d", 
    //                   first_store_dir, ssti->name.ptr(), store_dir, ssti->name.ptr(), err);
    //       }
    //       else
    //       {
    //         TBSYS_LOG(ERROR, "cp_safe(src_dir=%s, src_file=%s, dst_dir=%s, dst_file=%s)=>%d", 
    //                   first_store_dir, ssti->name.ptr(), store_dir, ssti->name.ptr(), err);
    //       }
    //     }
    //   }
    //   return err;
    // }
    
    // int fetch_sst_and_log_file_func(const volatile bool& stop, IObScpHandler* scp, const ObServer& master, const char* log_dir, StoreMgr& store_mgr,
    //                                 const ObUpsFetchParam& fetch_param, const int64_t timeout)
    // {
    //   int err = OB_SUCCESS;
    //   ObFileClient fs_client;
    //   TBSYS_LOG(INFO, "fetch_sst_and_log_file()");
    //   if (NULL == log_dir || NULL == scp)
    //   {
    //     err = OB_INVALID_ARGUMENT;
    //     TBSYS_LOG(ERROR, "fetch_sst_and_log_file_func(log_dir=%s, scp=%p)=>%d",
    //               log_dir, scp, err);
    //   }
    //   else if (OB_SUCCESS != (err = fetch_sst_file(stop, scp, master, store_mgr, fetch_param.sst_list_, timeout)))
    //   {
    //     TBSYS_LOG(ERROR, "fetch_sst_file(log_dir=%s)=>%d", log_dir, err);
    //   }
    //   else if (OB_SUCCESS != (err = fetch_log_file(stop, scp, master, log_dir, fetch_param.min_log_id_, fetch_param.max_log_id_, timeout)))
    //   {
    //     TBSYS_LOG(ERROR, "fetch_log_file(log_dir=%s, min_log_id=%ld, max_log_id=%ld)=>%d",
    //               log_dir, fetch_param.min_log_id_, fetch_param.max_log_id_, err);
    //   }
    //   return err;
    // }

    // int rsync_file(const ObServer& server, const char* remote_dir, const char* remote_file,
    //                const char* local_dir, const char* local_file, int64_t bwlimit)
    // {
    //   int err = OB_SUCCESS;
    //   int sys_err = 0;
    //   int len = 0;
    //   char cmd[256];
    //   char server_addr[128];
    //   char fetch_opt[128];
    //   const char* usr_opt = "";
    //   if (NULL == remote_dir || NULL == remote_file || NULL == local_dir || NULL == local_file)
    //   {
    //     err = OB_INVALID_ARGUMENT;
    //     TBSYS_LOG(ERROR, "cp_remote_file_func(remote_dir=%s, remote_file=%s, local_dir=%s, local_file=%s)=>%d",
    //               remote_dir, remote_file, local_dir, local_file, err);
    //   }
    //   else if (!server.ip_to_string(server_addr, sizeof(server_addr)))
    //   {
    //     err = OB_ERR_UNEXPECTED;
    //     TBSYS_LOG(ERROR, "server.to_string()=>false");
    //   }
    //   else if (0 > (len = snprintf(fetch_opt, sizeof(fetch_opt), "%s --bwlimit=%ld", usr_opt, bwlimit))
    //            || len > (int)sizeof(fetch_opt))
    //   {
    //     err = OB_BUF_NOT_ENOUGH;
    //     TBSYS_LOG(ERROR, "snprintf(fetch_option, usr_opt=%s, limit_rate=%ld, len=%ld)=>%d",
    //               usr_opt, bwlimit, sizeof(fetch_opt), len);
    //   }
    //   else if (0 > (len = snprintf(cmd, sizeof(cmd), "rsync %s %s:%s/%s %s/%s", fetch_opt, server_addr,
    //                               remote_dir, remote_file, local_dir, local_file))
    //            || len > (int)sizeof(cmd))
    //   {
    //     err = OB_BUF_NOT_ENOUGH;
    //     TBSYS_LOG(ERROR, "snprintf(fetch_opt=%s, addr=%s, remote_dir=%s, remote_file=%s, local_dir=%s, local_file=%s)=>%d",
    //               fetch_opt, server_addr, remote_dir, remote_file, local_dir, local_file, err);
    //   }
    //   else if (0 != (sys_err = FSU::vsystem(cmd)))
    //   {
    //     err = OB_ERR_SYS;
    //     TBSYS_LOG(ERROR, "vsystem('%s')=>[err=%d]", cmd, sys_err);
    //   }
    //   else
    //   {
    //     TBSYS_LOG(INFO, "vsystem('%s')=>OK", cmd);
    //   }
    //   return err;
    // }
    
    // int client_get_file(ObFileClient& client, const ObServer& server,
    //                         const char* remote_dir, const char* remote_file,
    //                         const char* local_dir, const char* local_file, const int64_t timeout)
    // {
    //   int err = OB_SUCCESS;
    //   char cbuf[OB_MAX_FILE_NAME_LENGTH];
    //   ObString remote_path_str;
    //   ObString local_dir_str;
    //   ObString local_file_str;
    //   if (NULL == remote_dir || NULL == remote_file || NULL == local_dir || NULL == local_file)
    //   {
    //     err = OB_INVALID_ARGUMENT;
    //     TBSYS_LOG(ERROR, "cp_remote_file_func(remote_dir=%s, remote_file=%s, local_dir=%s, local_file=%s)=>%d",
    //               remote_dir, remote_file, local_dir, local_file, err);
    //   }
    //   else if (OB_SUCCESS != (err = join_path(cbuf, sizeof(cbuf), remote_dir, remote_file)))
    //   {
    //     TBSYS_LOG(ERROR, "snprintf(buf_size=%d)=>%d", sizeof(cbuf), err);
    //   }
    //   else
    //   {
    //     remote_path_str.assign_ptr(cbuf, strnlen(cbuf, OB_MAX_FILE_NAME_LENGTH));
    //     local_dir_str.assign_ptr((char*)local_dir, strnlen(local_dir, OB_MAX_FILE_NAME_LENGTH));
    //     local_file_str.assign_ptr((char*)local_file, strnlen(local_file, OB_MAX_FILE_NAME_LENGTH));
    //   }

    //   if (OB_SUCCESS != err)
    //   {}
    //   else if(OB_SUCCESS != (err = client.get_file(timeout, server, remote_path_str, local_dir_str, local_file_str)))
    //   {
    //     TBSYS_LOG(ERROR, "client.get_file(remote_dir=%s, remote_file=%s, local_dir=%s, local_file=%s, timeout=%ld)=>%d",
    //               remote_dir, remote_file, local_dir, local_file, timeout, err);
    //   }
    //   return err;
    // }

    // int cp_remote_file_func(ObFileClient* client, const ObServer& server,
    //                         const char* remote_dir, const char* remote_file,
    //                         const char* local_dir, const char* local_file, const int64_t timeout)
    // {
    //   int err = OB_SUCCESS;
    //   char server_addr[128];
    //   server.to_string(server_addr, sizeof(server_addr));
    //   server_addr[sizeof(server_addr)-1] = 0;
    //   TBSYS_LOG(INFO, "cp_remote_file(%s:%s/%s => %s/%s)", server_addr, remote_dir, remote_file, local_dir, local_file);
    //   if(NULL != client)
    //   {
    //     err = client_get_file(*client, server, remote_dir,  remote_file, local_dir, local_file, timeout);
    //   }
    //   else
    //   {
    //     int64_t bwlimit = 10*1000;
    //     err = rsync_file(server, remote_dir, remote_file, local_dir, local_file, bwlimit);
    //   }
    //   return err;
    // }

    int replay_single_log_func(ObUpsMutator& mutator, CommonSchemaManagerWrapper& schema,
                               ObUpsTableMgr* table_mgr, LogCommand cmd, const char* log_data, int64_t data_len)
    {
      int err = OB_SUCCESS;
      int64_t pos = 0;
      int64_t log_id;
      if (NULL == table_mgr || NULL == log_data || 0 >= data_len)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "replay_single_log_func(table_mgr=%p, cmd=%d, log_data=%p, data_len=%ld)=>%d", table_mgr, cmd, log_data, data_len, err);
      }
      else
      {
        switch(cmd)
        {
          case OB_LOG_UPS_MUTATOR:
            if (OB_SUCCESS != (err = mutator.deserialize(log_data, data_len, pos)))
            {
              TBSYS_LOG(ERROR, "UpsMutator deserialize error[ret=%d log_data=%p data_len=%ld]", err, log_data, data_len);
            }
            else if (OB_SUCCESS != (err = table_mgr->replay(mutator)))
            {
              TBSYS_LOG(ERROR, "UpsTableMgr replay error[ret=%d]", err);
            }
            break;
          case OB_UPS_SWITCH_SCHEMA:
            if (OB_SUCCESS != (err = schema.deserialize(log_data, data_len, pos)))
            {
              TBSYS_LOG(ERROR, "ObSchemaManagerWrapper deserialize error[ret=%d log_data=%p data_len=%ld]",
                        err, log_data, data_len);
            }
            else if (OB_SUCCESS != (err = table_mgr->set_schemas(schema)))
            {
              TBSYS_LOG(ERROR, "UpsTableMgr set_schemas error, ret=%d schema_version=%ld", err, schema.get_version());
            }
            else
            {
              TBSYS_LOG(INFO, "switch schema succ");
            }
            break;
          case OB_LOG_SWITCH_LOG:
            if (OB_SUCCESS != (err = serialization::decode_i64(log_data, data_len, pos, (int64_t*)&log_id)))
            {
              TBSYS_LOG(ERROR, "decode_i64 log_id error, err=%d", err);
            }
            else
            {
              pos = data_len;
              TBSYS_LOG(INFO, "replay thread: SWITCH_LOG, log_id=%ld", log_id);
            }
            break;
          case OB_LOG_NOP:
            pos = data_len;
            break;
          default:
            err = OB_ERROR;
            break;
        }
      }
      if (pos != data_len)
      {
        TBSYS_LOG(ERROR, "pos[%ld] != data_len[%ld]", pos, data_len);
        err = OB_ERROR;
      }
      if (OB_SUCCESS != err)
      {
        TBSYS_LOG(ERROR, "replay_log(cmd=%d, log_data=%p, data_len=%ld)=>%d", cmd, log_data, data_len, err);
        common::hex_dump(log_data, data_len, false, TBSYS_LOG_LEVEL_WARN);
      }
      return err;
    }

    // int replay_log_func(const volatile bool& stop, const char* log_dir, const ObLogCursor& start_cursor, const ObLogCursor& end_cursor, ObLogCursor& return_cursor, IObLogApplier* log_applier)
    // {
    //   int err = OB_SUCCESS;
    //   int64_t retry_wait_time = 10 * 1000;
    //   ObLogCursor log_cursor = start_cursor;
    //   ObLogReader log_reader;
    //   ObDirectLogReader direct_reader;
    //   char* log_data = NULL;
    //   int64_t data_len = 0;
    //   LogCommand cmd = OB_LOG_UNKNOWN;
    //   uint64_t seq;

    //   if (NULL == log_dir || NULL == log_applier || log_cursor.file_id_ <= 0 || log_cursor.log_id_ != 0 || log_cursor.offset_ != 0)
    //   {
    //     err = OB_INVALID_ARGUMENT;
    //     TBSYS_LOG(ERROR, "invalid argument: log_dir=%s, log_applier=%p, log_cursor=%s", log_dir, log_applier, log_cursor.to_str());
    //   }
    //   else if (OB_SUCCESS != (err = log_reader.init(&direct_reader, log_dir, log_cursor.file_id_, 0, false)))
    //   {
    //     TBSYS_LOG(ERROR, "ObLogReader init error[err=%d]", err);
    //   }
    //   while (!stop && OB_SUCCESS == err)
    //   {
    //     //TBSYS_LOG(INFO, "log_cursor=%s", log_cursor.to_str());
    //     if (OB_SUCCESS != (err = log_reader.read_log(cmd, seq, log_data, data_len)) &&
    //         OB_FILE_NOT_EXIST != err && OB_READ_NOTHING != err)
    //     {
    //       TBSYS_LOG(ERROR, "ObLogReader read error[ret=%d]", err);
    //     }
    //     else if (OB_SUCCESS != err)
    //     {}
    //     else if (OB_SUCCESS != (err = log_reader.get_cursor(log_cursor)))
    //     {
    //       TBSYS_LOG(ERROR, "log_reader.get_cursor()=>%d",  err);
    //     }
    //     else if (end_cursor.is_valid()? !end_cursor.newer_than(log_cursor): (OB_SUCCESS != err))
    //     {
    //       err = OB_ITER_END;
    //     }
    //     else if (OB_FILE_NOT_EXIST == err || OB_READ_NOTHING == err || OB_LAST_LOG_RUINNED == err)
    //     {
    //       err = OB_ITER_END;
    //       usleep(retry_wait_time);
    //     }// replay all
    //     else if (OB_SUCCESS != (err = log_applier->apply_log(cmd, seq, log_data, data_len)))
    //     {
    //       TBSYS_LOG(ERROR, "replay_single_log()=>%d",  err);
    //     }
    //   }

    //   if (OB_ITER_END == err)
    //   {
    //     err = OB_SUCCESS;
    //   }
    //   if (OB_SUCCESS == err)
    //   {
    //     return_cursor = log_cursor;
    //   }
    //   TBSYS_LOG(INFO, "replay_log(log_dir=%s, start_cursor=%s)", log_dir, start_cursor.to_str());
    //   TBSYS_LOG(INFO, "replay_log(log_dir=%s, return_cursor=%s)", log_dir, return_cursor.to_str());
    //   return err;
    // }
    
    int replay_local_log_func(const volatile bool& stop, const char* log_dir, ObLogCursor& log_cursor, IObLogApplier* log_applier)
    {
      int err = OB_SUCCESS;
      ObLogReader log_reader;
      ObDirectLogReader direct_reader;
      char* log_data = NULL;
      int64_t data_len = 0;
      LogCommand cmd = OB_LOG_UNKNOWN;
      uint64_t seq;

      if (NULL == log_dir || NULL == log_applier || log_cursor.file_id_ <= 0 || log_cursor.log_id_ != 0 || log_cursor.offset_ != 0)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "invalid argument: log_dir=%s, log_applier=%p, log_cursor=%s", log_dir, log_applier, log_cursor.to_str());
      }
      else if (OB_SUCCESS != (err = log_reader.init(&direct_reader, log_dir, log_cursor.file_id_, 0, false)))
      {
        TBSYS_LOG(ERROR, "ObLogReader init error[err=%d]", err);
      }
      while (!stop && OB_SUCCESS == err)
      {
        //TBSYS_LOG(INFO, "log_cursor=%s", log_cursor.to_str());
        if (OB_SUCCESS != (err = log_reader.read_log(cmd, seq, log_data, data_len)) &&
            OB_FILE_NOT_EXIST != err && OB_READ_NOTHING != err && OB_LAST_LOG_RUINNED != err))
        {
          TBSYS_LOG(ERROR, "ObLogReader read error[ret=%d]", err);
        }
        else if (OB_SUCCESS != err)
        {
          err = OB_ITER_END; // replay all
        }
        else if (OB_SUCCESS != (err = log_reader.get_cursor(log_cursor)))
        {
          TBSYS_LOG(ERROR, "log_reader.get_cursor()=>%d",  err);
        }
        else if (OB_SUCCESS != (err = log_applier->apply_log(cmd, seq, log_data, data_len)))
        {
          TBSYS_LOG(ERROR, "replay_single_log()=>%d",  err);
        }
      }

      if (OB_ITER_END == err)
      {
        err = OB_SUCCESS;
      }
      TBSYS_LOG(INFO, "replay_log(log_dir=%s, log_cursor=%s)", log_dir, log_cursor.to_str());
      return err;
    }

    int replay_log_in_buf_func(const char* log_data, int64_t data_len, IObLogApplier* log_applier)
    {
      int err = OB_SUCCESS;
      ObLogEntry log_entry;
      int64_t pos = 0;
      while (OB_SUCCESS == err && pos < data_len)
      {
        if (OB_SUCCESS != (err = log_entry.deserialize(log_data, data_len, pos)))
        {
          TBSYS_LOG(ERROR, "log_entry.deserialize(log_data=%p, data_len=%ld, pos=%ld)=>%d", log_data, data_len, pos, err);
        }
        else if (OB_SUCCESS != (err = log_entry.check_data_integrity(log_data + pos)))
        {
          TBSYS_LOG(ERROR, "log_entry.check_data_integrity()=>%d", err);
        }
        else if (OB_SUCCESS != (err = log_applier->apply_log((LogCommand)log_entry.cmd_, log_entry.seq_, log_data + pos, log_entry.get_log_data_len())))
        {
          TBSYS_LOG(ERROR, "replay_log(cmd=%d, log_data=%p, data_len=%ld)=>%d", log_entry.cmd_, log_data + pos, log_entry.get_log_data_len(), err);
        }
        else
        {
          pos += log_entry.get_log_data_len();
        }
      }
      return err;
    }

    int serialize_log_entry(char* buf, const int64_t len, int64_t& pos, ObLogEntry& entry,
                            const char* log_data, const int64_t data_len)
    {
      int err = OB_SUCCESS;
      if (NULL == buf || 0 >= len || pos > len || NULL == log_data || 0 >= data_len)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "serialize_log_entry(buf=%p, len=%ld, pos=%ld, log_data=%p, data_len=%ld)=>%d",
                  buf, len, pos, log_data, data_len, err);
      }
      else if (pos + entry.get_serialize_size() + data_len > len)
      {
        err = OB_BUF_NOT_ENOUGH;
        TBSYS_LOG(DEBUG, "pos[%ld] + entry.serialize_size[%ld] + data_len[%ld] > len[%ld]",
                  pos, entry.get_serialize_size(), data_len, len);
      }
      else if (OB_SUCCESS != (err = entry.serialize(buf, len, pos)))
      {
        TBSYS_LOG(ERROR, "entry.serialize(buf=%p, pos=%ld, capacity=%ld)=>%d",
                  buf, len, pos, err);
      }
      else
      {
        memcpy(buf + pos, log_data, data_len);
        pos += data_len;
      }
      return err;
    }

    int generate_log(char* buf, const int64_t len, int64_t& pos, ObLogCursor& cursor, const LogCommand cmd,
                 const char* log_data, const int64_t data_len)
    {
      int err = OB_SUCCESS;
      ObLogEntry entry;
      if (NULL == buf || 0 >= len || pos > len || NULL == log_data || 0 >= data_len || !cursor.is_valid())
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "generate_log(buf=%p, len=%ld, pos=%ld, log_data=%p, data_len=%ld, cursor=%s)=>%d",
                  buf, len, pos, log_data, data_len, cursor.to_str(), err);
      }
      else if (OB_SUCCESS != (err = cursor.next_entry(entry, cmd, log_data, data_len)))
      {
        TBSYS_LOG(ERROR, "cursor[%s].next_entry()=>%d", cursor.to_str(), err);
      }
      else if (OB_SUCCESS != (err = serialize_log_entry(buf, len, pos, entry, log_data, data_len)))
      {
        TBSYS_LOG(DEBUG, "serialize_log_entry(buf=%p, len=%ld, entry[id=%ld], data_len=%ld)=>%d",
                  buf, len, entry.seq_, data_len, err);
      }
      else if (OB_SUCCESS != (err = cursor.advance(entry)))
      {
        TBSYS_LOG(ERROR, "cursor[id=%ld].advance(entry.id=%ld)=>%d", cursor.log_id_, entry.seq_, err);
      }
      return err;
    }

    int set_entry(ObLogEntry& entry, const int64_t seq, const LogCommand cmd, const char* log_data, const int64_t data_len)
    {
      int err = OB_SUCCESS;
      entry.set_log_seq(seq);
      entry.set_log_command(cmd);
      if (OB_SUCCESS != (err = entry.fill_header(log_data, data_len)))
      {
        TBSYS_LOG(ERROR, "entry.fill_header(log_data=%p, data_len=%ld)=>%d", log_data, data_len, err);
      }
      return err;
    }

    int serialize_log_entry(char* buf, const int64_t len, int64_t& pos, const LogCommand cmd, const int64_t seq, const char* log_data, const int64_t data_len)
    {
      int err = OB_SUCCESS;
      ObLogEntry entry;
      if (OB_SUCCESS != (err = set_entry(entry, seq, cmd, log_data, data_len)))
      {
        TBSYS_LOG(ERROR, "set_entry(seq=%ld, cmd=%d, log_data=%p, data_len=%ld)=>%d", seq, cmd, log_data, data_len, err);
      }
      else if (OB_SUCCESS != (err = serialize_log_entry(buf, len, pos, entry, log_data, data_len)))
      {
        TBSYS_LOG(ERROR, "serialize_log_entry(buf=%p, len=%ld, pos=%ld, log_data=%p, data_len=%ld)=>%d",
                  buf, len, pos, log_data, data_len, err);
      }
      return err;
    }

    int fill_log(char* buf, const int64_t len, int64_t& pos, ObLogCursor& cursor, const LogCommand cmd, const char* log_data, const int64_t data_len)
    {
      int err = OB_SUCCESS;
      ObLogEntry entry;
      if (NULL == buf || len < 0 || pos > len || NULL == log_data || data_len <= 0)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else if (OB_SUCCESS != (err = cursor.this_entry(entry, cmd, log_data, data_len)))
      {
        TBSYS_LOG(ERROR, "cursor[%s].next_entry()=>%d", cursor.to_str(), err);
      }
      else if (OB_SUCCESS != (err = serialize_log_entry(buf, len, pos, entry, log_data, data_len)))
      {
        TBSYS_LOG(DEBUG, "serialize_log_entry(buf[remain=%ld], entry[id=%ld], data_len=%ld)=>%d",
                  len-pos, entry.seq_, data_len, err);
      }
      else if (OB_SUCCESS != (err = cursor.advance(entry)))
      {
        TBSYS_LOG(ERROR, "cursor[id=%ld].advance(entry.id=%ld)=>%d", cursor.log_id_, entry.seq_, err);
      }
      return err;
    }

    // int require_log_reader_seek_to(ObSingleLogReader* log_reader, const common::ObLogCursor& cursor)
    // {
    //   int err = OB_SUCCESS;
    //   ObLogCursor real_cursor;
    //   if (NULL == log_reader || !cursor.is_valid())
    //   {
    //     err = OB_INVALID_ARGUMENT;
    //     TBSYS_LOG(ERROR, "log_reader_seek_to(log_reader=%p, cursor=%s): invalid argument.",
    //               log_reader, cursor.to_str());
    //   }
    //   else if (OB_SUCCESS != (err = log_reader->get_cursor(real_cursor)))
    //   {
    //     TBSYS_LOG(ERROR, "log_reader.get_cursor()=>%d", err);
    //   }
    //   else if (cursor.equal(real_cursor))
    //   {}
    //   else if (OB_SUCCESS != (err = log_reader->close()))
    //   {
    //     TBSYS_LOG(ERROR, "log_reader.close()=>%d", err);
    //   }
    //   else if (OB_SUCCESS != (err = log_reader->open(cursor.file_id_, cursor.log_id_, cursor.offset_)) &&
    //            OB_FILE_NOT_EXIST != err)
    //   {
    //     TBSYS_LOG(ERROR, "log_reader.open(%s)=>%d", cursor.to_str(), err);
    //   }
    //   else if (OB_FILE_NOT_EXIST == err)
    //   {
    //     TBSYS_LOG(INFO, "log_reader.open(%s)=>%d", cursor.to_str(), err);
    //     err = OB_READ_NOTHING;
    //   }

    //   return err;
    // }

    // int read_single_log(char* buf, const int64_t len, int64_t& pos, ObRepeatedLogReader& log_reader, ObLogCursor& log_cursor)
    // {
    //   int err = OB_SUCCESS;
    //   LogCommand cmd = OB_LOG_UNKNOWN;
    //   int64_t seq = 0;
    //   char* log_data = NULL;
    //   int64_t data_len = 0;
    //   ObLogCursor tmp_cursor;
    //   if (OB_SUCCESS != (err = log_reader.get_cursor(tmp_cursor)))
    //   {
    //     TBSYS_LOG(ERROR, "log_reader.get_cursor()=>%d", err);
    //   }
    //   else if (OB_SUCCESS != (err = log_reader.read_log(cmd, (uint64_t&)seq, log_data, data_len)))
    //   {
    //     if (OB_READ_NOTHING != err && OB_FILE_NOT_EXIST != err)
    //     {
    //       TBSYS_LOG(ERROR, "log_reader.read_log()=>%d", err);
    //     }
    //   }
    //   else if (OB_SUCCESS != (err = serialize_log_entry(buf, len, pos, cmd, seq, (const char*)log_data, data_len)))
    //   {
    //     TBSYS_LOG(INFO, "serialize_log_entry(buf=%p, len=%ld, pos=%ld, cmd=%d, seq=%ld, log_data=%p, data_len=%ld, buf.capacity=%ld)=>%d",
    //               buf, len, pos, cmd, seq, log_data, data_len, err);
    //   }
    //   else if (OB_LOG_SWITCH_LOG == cmd)
    //   {
    //     TBSYS_LOG(INFO, "read switch log(file_id=%ld, log_id=%ld, offset=%ld)",
    //               log_cursor.file_id_, log_cursor.log_id_, log_cursor.offset_);
    //     tmp_cursor.file_id_ ++;
    //     tmp_cursor.log_id_ ++;
    //     tmp_cursor.offset_ = 0;
    //   }
    //   else if (OB_SUCCESS != (err = log_reader.get_cursor(tmp_cursor)))
    //   {
    //     TBSYS_LOG(ERROR, "log_reader.get_cursor()=>%d", err);
    //   }

    //   if (OB_SUCCESS == err)
    //   {
    //     log_cursor = tmp_cursor;
    //   }
    //   if (OB_BUF_NOT_ENOUGH == err || OB_FILE_NOT_EXIST == err)
    //   {
    //     err = OB_READ_NOTHING;
    //   }
    //   return err;
    // }

    // int read_multiple_logs(char* buf, const int64_t len, int64_t& pos, ObRepeatedLogReader& log_reader,
    //                        const ObLogCursor& start_cursor, ObLogCursor& end_cursor)
    // {
    //   int err = OB_SUCCESS;
    //   int64_t old_pos = pos;
    //   end_cursor = start_cursor;
    //   while(OB_SUCCESS == err)
    //   {
    //     err = read_single_log(buf, len, pos, log_reader, end_cursor);
    //     if (OB_SUCCESS != err && OB_READ_NOTHING != err)
    //     {
    //       TBSYS_LOG(ERROR, "read_single_log()=>%d", err);
    //     }
    //   }

    //   if (OB_SUCCESS == err && old_pos == pos)
    //   {
    //     err = OB_READ_NOTHING;
    //   }
    //   if (old_pos < pos && OB_READ_NOTHING == err)
    //   {
    //     err = OB_SUCCESS;
    //   }
    //   return err;
    // }
  int parse_log_buffer(const char* log_data, const int64_t len, int64_t& start_id, int64_t& end_id)
    {
      int err = OB_SUCCESS;
      int64_t end_pos = 0;
      if (OB_SUCCESS != (err = trim_log_buffer(log_data, len, end_pos, start_id, end_id)))
      {
        TBSYS_LOG(ERROR, "trim_log_buffer(log_data=%p[%ld])=>%d", log_data, len, err);
      }
      else if (end_pos != len)
      {
        err = OB_INVALID_LOG;
      }
      return err;
    }
    int trim_log_buffer(const char* log_data, const int64_t len, int64_t& end_pos,
                        int64_t& start_id, int64_t& end_id)
    {
      bool is_file_end = false;
      return trim_log_buffer(log_data, len, end_pos, start_id, end_id, is_file_end);
    }

    int trim_log_buffer(const char* log_data, const int64_t len, int64_t& end_pos,
                        int64_t& start_id, int64_t& end_id, bool& is_file_end)
    {
      int err = OB_SUCCESS;
      int64_t pos = 0;
      int64_t old_pos = 0;
      ObLogEntry log_entry;
      int64_t real_start_id = 0;
      int64_t real_end_id = 0;
      end_pos = 0;
      if (NULL == log_data || 0 > len)
      {
        err = OB_INVALID_ARGUMENT;
        TBSYS_LOG(ERROR, "trim_log_buffer(buf=%p[%ld]): invalid argument", log_data, len);
      }
      while (OB_SUCCESS == err && pos + log_entry.get_serialize_size() < len)
      {
        old_pos = pos;
        if (OB_SUCCESS != (err = log_entry.deserialize(log_data, len, pos)))
        {
          TBSYS_LOG(ERROR, "log_entry.deserialize(log_data=%p, len=%ld, pos=%ld)=>%d", log_data, len, pos, err);
        }
        else if (old_pos + log_entry.get_serialize_size() + log_entry.get_log_data_len() > len)
        {
          pos = old_pos;
          break;
        }
        else if (OB_SUCCESS != (err = log_entry.check_data_integrity(log_data + pos)))
        {
          TBSYS_LOG(ERROR, "log_entry.check_data_integrity()=>%d", err);
        }

        if (OB_SUCCESS != err)
        {}
        else if (real_end_id > 0 && real_end_id != (int64_t)log_entry.seq_)
        {
          err = OB_DISCONTINUOUS_LOG;
        }
        else
        {
          pos = old_pos + log_entry.get_serialize_size() + log_entry.get_log_data_len();
          real_end_id = log_entry.seq_ + 1;
          if (0 >= real_start_id)
          {
            real_start_id = log_entry.seq_;
          }
          if (OB_LOG_SWITCH_LOG == log_entry.cmd_)
          {
            is_file_end = true;
            break;
          }
        }
      }

      if (OB_SUCCESS != err && OB_INVALID_ARGUMENT != err)
      {
        TBSYS_LOG(ERROR, "parse log buf error:");
        hex_dump(log_data, len, true, TBSYS_LOG_LEVEL_WARN);
      }
      else if (real_start_id > 0)
      {
        end_pos = pos;
        start_id = real_start_id;
        end_id = real_end_id;
      }
      return err;
    }
  } // end namespace updateserver
} // end namespace oceanbase
