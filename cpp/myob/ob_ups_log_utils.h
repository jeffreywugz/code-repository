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

#ifndef __OB_UPDATESERVER_OB_UPS_LOG_UTILS_H__
#define __OB_UPDATESERVER_OB_UPS_LOG_UTILS_H__

#include "common/ob_define.h"
#include "common/ob_log_entry.h"
#include "common/ob_log_cursor.h"

namespace oceanbase
{
  namespace common
  {
    class ObSingleLogReader;
    class ObServer;
    class ObFileClient;
    class ObRepeatedLogReader;
  }
  namespace updateserver
  {
    class IErrorReporter
    {
      public:
        IErrorReporter(){}
        virtual ~IErrorReporter(){}
        virtual int report_error(int err_code) = 0;
    };

    class IObLogApplier
    {
      public:
        IObLogApplier(){}      
        virtual ~IObLogApplier(){}
        virtual int apply_log(const common::LogCommand cmd, const int64_t seq,
                              const char* log_data, const int64_t data_len) = 0;
    };

    class ObUpsTableMgr;
    class ObUpsMutator;
    class CommonSchemaManagerWrapper;
    int load_replay_point_func(const char* log_dir, int64_t& replay_point);
    int write_replay_point_func(const char* log_dir, const int64_t replay_point);
    int scan_log_dir_func(const char* log_dir, int64_t& replay_point, int64_t& min_log_id, int64_t& max_log_id);
    int get_local_max_log_cursor_func(const char* log_dir, ObLogCursor& log_cursor);
    int get_replay_point_func(const char* log_dir, int64_t& replay_point);
    // int seek_log_by_id(int64_t seq_id, int64_t& n_skiped, const char* log_data, int64_t data_len, int64_t& pos);
    //int crop_log_task(ObBatchLogEntryTask* tasks, int64_t seq_id);
    int parse_log_buffer(const char* log_data, const int64_t len, const int64_t limit, int64_t& end_pos,
                         const common::ObLogCursor& start_cursor, common::ObLogCursor& end_cursor);
    int logcpy(char* dest, const char* src, const int64_t len, const int64_t limit, int64_t bytes_cpy,
               const common::ObLogCursor& start_cursor, common::ObLogCursor& end_cursor);
    int join_path(char* result_path, const int64_t len_limit, const char* base, const char* path);
    int get_absolute_path(char* abs_path, const int64_t len_limit, const char* path);
    int rsync_file(const common::ObServer& server, const char* remote_dir, const char* remote_file,
                   const char* local_dir, const char* local_file, int64_t bwlimit);
    // int client_get_file(common::ObFileClient& client, const common::ObServer& server,
    //                         const char* remote_dir, const char* remote_file,
    //                     const char* local_dir, const char* local_file, const int64_t timeout);
    // int cp_remote_file_func(common::ObFileClient* client, const common::ObServer& server,
    //                         const char* remote_dir, const char* remote_file,
    //                         const char* local_dir, const char* local_file, const int64_t timeout);
    // int fetch_sst_and_log_file_func(const volatile bool& stop, IObScpHandler* scp_handler,
    //                                 const common::ObServer& master, const char* log_dir, StoreMgr& store_mgr,
    //                                 const ObUpsFetchParam& fetch_param, int64_t timeout);
    int replay_single_log_func(ObUpsMutator& mutator, CommonSchemaManagerWrapper& schema, ObUpsTableMgr* table_mgr, common::LogCommand cmd, const char* log_data, int64_t data_len);
    int replay_log_func(const volatile bool& stop, const char* log_dir,
                        const common::ObLogCursor& start_cursor, const common::ObLogCursor& end_cursor,
                        common::ObLogCursor& return_cursor, IObLogApplier* log_applier);
    int replay_local_log_func(const volatile bool& stop, const char* log_dir, ObLogCursor& log_cursor, IObLogApplier* log_applier);
    int replay_log_in_buf_func(const char* log_data, int64_t data_len, IObLogApplier* log_applier);

    int serialize_log_entry(char* buf, const int64_t len, int64_t& pos, common::ObLogEntry& entry,
                            const char* log_data, const int64_t data_len);
    int generate_log(char* buf, const int64_t len, int64_t& pos, common::ObLogCursor& cursor, const common::LogCommand cmd,
                     const char* log_data, const int64_t data_len);
    int set_entry(common::ObLogEntry& entry, const int64_t seq, const common::LogCommand cmd,
                  const char* log_data, const int64_t data_len);
    int serialize_log_entry(char* buf, const int64_t len, int64_t& pos, const common::LogCommand cmd, const uint64_t seq,
                            const char* log_data, const int64_t data_len);
    int require_log_reader_seek_to(common::ObSingleLogReader* log_reader, const common::ObLogCursor& cursor);
    int read_single_log(char* buf, const int64_t len, int64_t& pos,
                        common::ObRepeatedLogReader& log_reader, common::ObLogCursor& log_cursor);
    int read_multiple_logs(char* buf, const int64_t len, int64_t& pos, common::ObRepeatedLogReader& log_reader,
                           const common::ObLogCursor& start_cursor, common::ObLogCursor& end_cursor);
    int parse_log_buffer(const char* log_data, const int64_t len,
                         int64_t& start_id, int64_t& end_id);
    int trim_log_buffer(const char* log_data, const int64_t len, int64_t& end_pos,
                        int64_t& start_id, int64_t& end_id);
    int trim_log_buffer(const char* log_data, const int64_t len, int64_t& end_pos,
                        int64_t& start_id, int64_t& end_id, bool& is_file_end);
 } // end namespace updateserver
} // end namespace oceanbase
#endif /* __OB_UPDATESERVER_OB_UPS_LOG_UTILS_H__ */
