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

#ifndef __OB_UPDATESERVER_OB_DEFAULT_LOG_LOCATOR_H__
#define __OB_UPDATESERVER_OB_DEFAULT_LOG_LOCATOR_H__

#include "ob_log_locator.h"

namespace oceanbase
{
  namespace updateserver
  {
//
    // 根据日志ID查询文件ID
    class IObLogFileIdLocator
    {
      public:
        IObLogFileIdLocator(){}
        virtual ~IObLogFileIdLocator(){}
        virtual int get_file_id(const int64_t log_id, int64_t& file_id) = 0;
    };

    // 根据日志ID和文件ID查询文件偏移
    class IObLogFileOffsetLocator
    {
      public:
        IObLogFileOffsetLocator(){}
        virtual ~IObLogFileOffsetLocator(){}
        virtual int get_file_offset(const int64_t log_id, const int64_t file_id, int64_t& offset) = 0;
    };

    // 通过扫描磁盘文件定位日志
    class ObOnDiskLogLocator : public IObLogLocator
    {
      public:
        ObOnDiskLogLocator() : file_id_locator_(NULL), file_offset_locator_(NULL) {}
        virtual ~ObOnDiskLogLocator(){}
        bool is_inited() const;
        int init(IObLogFileIdLocator* file_id_locator, IObLogFileOffsetLocator* file_offset_locator);
        virtual int get_location(const int64_t log_id, ObLogLocation& location);
        virtual int enter_location(const ObLogLocation& location);
      private:
        IObLogFileIdLocator* file_id_locator_;
        IObLogFileOffsetLocator* file_offset_locator_;
    };

    class IObFirstLogIdGetter
    {
      public:
        IObFirstLogIdGetter(){}
        virtual ~IObFirstLogIdGetter(){}
        virtual int get_first_log_id(const int64_t file_id, int64_t& log_id) = 0;
    };

    class IObMaxLogFileIdGetter
    {
      public:
        IObMaxLogFileIdGetter(){}
        virtual ~IObMaxLogFileIdGetter(){}
        virtual int get_max_log_file_id(int64_t& max_log_file_id) = 0;
    };

    class ObLogFileIdLocator : public IObLogFileIdLocator
    {
      public:
        ObLogFileIdLocator();
        virtual ~ObLogFileIdLocator();
        bool is_inited() const;
        int init(IObMaxLogFileIdGetter* max_log_file_id_getter,
                 IObFirstLogIdGetter* first_log_id_getter, const int64_t n_indexes);
        // max_log_file_id 表示当前最大的日志文件编号，可以从log_mgr获得。
        int get_file_id_helper(const int64_t max_log_file_id, int64_t log_id, int64_t& file_id);
        virtual int get_file_id(int64_t log_id, int64_t& file_id);
        int get_first_log_id(int64_t file_id, int64_t& start_log_id);
      private:
        IObMaxLogFileIdGetter* max_log_file_id_getter_;
        IObFirstLogIdGetter* first_log_id_getter_;
        ObRecentCache<int64_t, int64_t> first_log_id_cache_;
    };

    class ObFirstLogIdGetter : public IObFirstLogIdGetter
    {
      public:
        ObFirstLogIdGetter();
        virtual ~ObFirstLogIdGetter();
        int init(const char* log_dir);
        virtual int get_first_log_id(const int64_t file_id, int64_t& log_id);
      private:
        const char* log_dir_;
    };

    class ObLogFileOffsetLocator : public IObLogFileOffsetLocator
    {
      public:
        ObLogFileOffsetLocator();
        virtual ~ObLogFileOffsetLocator();
        int init(const char* log_dir);
        virtual int get_file_offset(const int64_t log_id, const int64_t file_id, int64_t& offset);
      private:
        const char* log_dir_;
    };

    class ObStoredMaxLogFileIdGetter : public IObMaxLogFileIdGetter
    {
      public:
        ObStoredMaxLogFileIdGetter(): max_log_file_id_(0) {}        
        ~ObStoredMaxLogFileIdGetter() {}
        virtual int get_max_log_file_id(int64_t& max_log_file_id);
        int set_max_log_file_id(const int64_t max_log_file_id);
      private:
        int64_t max_log_file_id_;
    };

    class ObDefaultLogLocator : public IObLogLocator
    {
      public:
        ObDefaultLogLocator();
        virtual ~ObDefaultLogLocator();
        bool is_inited() const;
        int init(const char* log_dir, IObMaxLogFileIdGetter* max_log_file_id_getter,
                 const int64_t file_id_cache_capacity, const int64_t log_id_cache_capacity);
        virtual int get_location(const int64_t log_id, ObLogLocation& location);
        virtual int enter_location(const ObLogLocation& location);
        int set_max_log_file_id(const int64_t max_log_file_id);
      private:
        const char* log_dir_;
        ObCachedLogLocator cached_log_locator_;
        ObLogLocationCache log_location_cache_;
        ObOnDiskLogLocator on_disk_log_locator_;
        ObFirstLogIdGetter first_log_id_getter_;
        ObLogFileIdLocator file_id_locator_;
        ObLogFileOffsetLocator offset_locator_;
    };
  }; // end namespace updateserver
}; // end namespace oceanbase
#endif /* __OB_UPDATESERVER_OB_DEFAULT_LOG_LOCATOR_H__ */
