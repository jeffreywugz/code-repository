/*
 * (C) 2007-2010 Taobao Inc.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 *
 * Authors:
 *   yuanqi.xhf <yuanqi.xhf@taobao.com>
 *     - some work details if you want
 *
 */

#ifndef __OCEANBASE_UPDATE_SERVER_OB_RECENT_LOG_CACHE_H__
#define __OCEANBASE_UPDATE_SERVER_OB_RECENT_LOG_CACHE_H__
#include "common/ob_define.h"
#include "common/ob_log_cursor.h"

namespace oceanbase
{
  namespace updateserver
  {
    struct ObBufferedLogFileIndex
    {
      ObBufferedLogFileIndex(): file_id_(-1), bytes_written_before_(0),
                                bytes_written_after_(0), start_file_offset_(0),
                                start_log_id_(0), end_log_id_(0) {}
      ~ObBufferedLogFileIndex() {}
      volatile int64_t file_id_;
      volatile uint64_t bytes_written_before_;
      volatile uint64_t bytes_written_after_;
      int64_t start_file_offset_; // 追加日志文件并不一定是从文件起始位置开始，
      // start_file_offset_表示缓冲区中该文件最开始的日志在文件内的偏移。
      int64_t start_log_id_;
      int64_t end_log_id_;
    };

    class ObRecentLogCache: public ObAtomicLogCursor
    {
      friend class ObRecentLogCacheTest;
      public:
        const static int64_t N_LOG_FILE_INDEX_ENTRY = 1<<4;
        ObRecentLogCache();
        ~ObRecentLogCache();
        bool is_inited() const;
        int check_state() const;
        int dump_for_debug() const;

        int init(const int64_t n_index_entries, ObBufferedLogFileIndex* const indexes, const int64_t log_buf_len,
                 char* const log_buf);
        int reset(); // reset()期间不能调用push()
        // push()只能被串行的调用，需要确保cursor连续
        // 如果cursor不连续，返回OB_DISCONTINUOUS_LOG, 之后需要调用reset()重置ObRecentLogCache
        int push_log(const common::ObLogCursor& start_cursor, const common::ObLogCursor& end_cursor,
                     const char* buf, int64_t len);
        // get()可以和push()并发调用, 可能返回OB_READ_NOTHING
        int get_log(const common::ObLogCursor& start_cursor, common::ObLogCursor& end_cursor,
                    char* buf, const int64_t len, int64_t& read_count);
        int get_index(const int64_t file_id, ObBufferedLogFileIndex& meta) const;
      protected:
        int check_cursor(const common::ObLogCursor& cursor) const;
  // fill_index()和get_mutable_index()只会被push_log()调用，执行期间不会有别的线程修改缓冲区状态
        int fill_index(const common::ObLogCursor& cursor);
        int get_mutable_index(const int64_t file_id, ObBufferedLogFileIndex*& meta) const;
        int get_log_const(const common::ObLogCursor& start_cursor, common::ObLogCursor& end_cursor,
                          char* buf, const int64_t len, int64_t& read_count) const;
      private:
        volatile int64_t total_bytes_written_;
        volatile int64_t start_bytes_written_;
        int64_t log_buffer_len_;
        char* log_buffer_;
        int64_t n_index_entries_;
        ObBufferedLogFileIndex* indexes_;
    };
    int init_log_cache_by_alloc(ObRecentLogCache& log_cache, int64_t n_index_entries, const int64_t log_buf_len,
                                char*& cbuf);
  } // end namespace updateserver
} // end namespace oceanbase
#endif //__OCEANBASE_UPDATE_SERVER_OB_RECENT_LOG_CACHE_H__
