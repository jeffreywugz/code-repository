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

#include "common/ob_define.h"
#include "common/ob_direct_log_reader.h"
#include "common/ob_log_dir_scanner.h"
#include "ob_default_log_locator.h"

using namespace oceanbase::common;
namespace oceanbase
{
  namespace updateserver
  {
    int get_max_log_file_id_in_dir(const char* log_dir, int64_t& max_log_file_id)
    {
      int err = OB_SUCCESS;
      ObLogDirScanner log_dir_scanner;
      if (NULL == log_dir)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else if (OB_SUCCESS != (err = log_dir_scanner.init(log_dir)))
      {
        TBSYS_LOG(ERROR, "log_dir_scanner.init(%s)=>%d", log_dir, err);
      }
      else if (OB_SUCCESS != (err = log_dir_scanner.get_max_log_id((uint64_t&)max_log_file_id)))
      {
        TBSYS_LOG(ERROR, "log_dir_scanner.get_max_log_id()=>%d", err);
      }
      return err;
    }

    ObDefaultLogLocator::ObDefaultLogLocator(): log_dir_(NULL)
    {}

    ObDefaultLogLocator::~ObDefaultLogLocator()
    {}

    bool ObDefaultLogLocator::is_inited() const
    {
      return NULL != log_dir_;
    }

    int ObDefaultLogLocator::init(const char* log_dir, IObMaxLogFileIdGetter* max_log_file_id_getter,
                                  const int64_t file_id_cache_capacity, const int64_t log_id_cache_capacity)
    {
      int err = OB_SUCCESS;
      if (is_inited())
      {
        err = OB_INIT_TWICE;
      }
      else if (NULL == log_dir || NULL == max_log_file_id_getter || 0 > file_id_cache_capacity || 0 > log_id_cache_capacity)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else if (OB_SUCCESS != (err = first_log_id_getter_.init(log_dir)))
      {
        TBSYS_LOG(ERROR, "first_log_id_getter_.init(log_dir=%s)=>%d", log_dir, err);
      }
      else if (OB_SUCCESS != (err = file_id_locator_.init(max_log_file_id_getter, &first_log_id_getter_, file_id_cache_capacity)))
      {
        TBSYS_LOG(ERROR, "file_id_locator_.init(cache_capacity=%ld)=>%d", file_id_cache_capacity, err);
      }
      else if (OB_SUCCESS != (err = offset_locator_.init(log_dir)))
      {
        TBSYS_LOG(ERROR, "offset_locator_.init(log_dir=%s)=>%d", log_dir, err);
      }
      else if (OB_SUCCESS != (err = on_disk_log_locator_.init(&file_id_locator_, &offset_locator_)))
      {
        TBSYS_LOG(ERROR, "on_disk_log_locator_.init(file_id_locator=%p, offset_locator=%p)=>%d",
                  &file_id_locator_, &offset_locator_, err);
      }
      else if (OB_SUCCESS != (err = log_location_cache_.init(log_id_cache_capacity)))
      {
        TBSYS_LOG(ERROR, "log_location_cache_.init(log_id_cache_capacity=%ld)=>%d",
                  log_id_cache_capacity, err);
      }
      else if (OB_SUCCESS != (err = cached_log_locator_.init(&log_location_cache_, &on_disk_log_locator_)))
      {
        TBSYS_LOG(ERROR, "cached_log_locator(cache=%p, backup=%p)=>%d",
                  &log_location_cache_, &on_disk_log_locator_, err);
      }
      else
      {
        log_dir_ = log_dir;
      }
      return err;
    }

    int ObDefaultLogLocator::get_location(const int64_t log_id, ObLogLocation& location)
    {
      return cached_log_locator_.get_location(log_id, location);
    }

    int ObDefaultLogLocator::enter_location(const ObLogLocation& location)
    {
      return cached_log_locator_.enter_location(location);
    }

    bool ObOnDiskLogLocator::is_inited() const
    {
      return NULL != file_id_locator_ && NULL != file_offset_locator_;
    }

    int ObOnDiskLogLocator::init(IObLogFileIdLocator* file_id_locator, IObLogFileOffsetLocator* file_offset_locator)
    {
      int err = OB_SUCCESS;
      if (is_inited())
      {
        err = OB_INIT_TWICE;
      }
      else if (NULL == file_id_locator || NULL == file_offset_locator)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else
      {
        file_id_locator_ = file_id_locator;
        file_offset_locator_ = file_offset_locator;
      }
      return err;
    }

    int ObOnDiskLogLocator::get_location(const int64_t log_id, ObLogLocation& location)
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (OB_SUCCESS != (err = file_id_locator_->get_file_id(log_id, location.file_id_)))
      {
        if (OB_ENTRY_NOT_EXIST != err)
        {
          TBSYS_LOG(ERROR, "get_file_id(log_id=%ld)=>%d", log_id, err);
        }
      }
      else if (OB_SUCCESS != (err = file_offset_locator_->get_file_offset(log_id, location.file_id_, location.offset_)))
      {
        if (OB_ENTRY_NOT_EXIST != err)
        {
          TBSYS_LOG(ERROR, "get_file_offset(log_id=%ld, file_id=%ld)=>%d", log_id, location.file_id_, err);
        }
      }
      return err;
    }

    int ObOnDiskLogLocator::enter_location(const ObLogLocation& location)
    {
      UNUSED(location);
      return OB_SUCCESS;
    }

    int get_first_log_id_func(const char* log_dir, const int64_t file_id, int64_t& log_id)
    {
      int err = OB_SUCCESS;
      ObDirectLogReader log_reader;
      LogCommand cmd = OB_LOG_NOP;
      uint64_t log_seq = 0;
      char* log_data = NULL;
      int64_t data_len = 0;

      if (NULL == log_dir || 0 >= file_id)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else if (OB_SUCCESS != (err = log_reader.init(log_dir)))
      {
        TBSYS_LOG(ERROR, "log_reader.init(log_dir=%s)=>%d", log_dir, err);
      }
      else if (OB_SUCCESS != (err = log_reader.open(file_id)))
      {
        if (OB_FILE_NOT_EXIST != err)
        {
          TBSYS_LOG(ERROR, "log_reader.open(file_id=%ld)=>%d", file_id, err);
        }
        else
        {
          err = OB_ENTRY_NOT_EXIST;
          TBSYS_LOG(DEBUG, "log_reader.open(file_id=%ld)=>FILE_NOT_EXIST", file_id);
        }
      }
      else if (OB_SUCCESS != (err = log_reader.read_log(cmd, log_seq, log_data, data_len)))
      {
        if (OB_READ_NOTHING != err)
        {
          TBSYS_LOG(ERROR, "log_reader.read_log()=>%d", err);
        }
        else
        {
          err = OB_ENTRY_NOT_EXIST;
          TBSYS_LOG(DEBUG, "log_reader.read_log(file_id=%ld)=>READ_NOTHING", file_id);
        }
      }
      else
      {
        log_id = (int64_t)log_seq;
      }
      return err;
    }

    int get_log_file_offset_func(const char* log_dir, const int64_t file_id, const int64_t log_id, int64_t& offset)
    {
      int err = OB_SUCCESS;
      ObDirectLogReader log_reader;
      LogCommand cmd = OB_LOG_NOP;
      uint64_t log_seq = 0;
      char* log_data = NULL;
      int64_t data_len = 0;
      if (NULL == log_dir || 0 >= file_id)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else if (OB_SUCCESS != (err = log_reader.init(log_dir)))
      {
        TBSYS_LOG(ERROR, "log_reader.init(log_dir=%s)=>%d", log_dir, err);
      }
      else if (OB_SUCCESS != (err = log_reader.open(file_id)))
      {
        if (OB_FILE_NOT_EXIST != err)
        {
          TBSYS_LOG(ERROR, "log_reader.open(file_id=%ld)=>%d", file_id, err);
        }
        else
        {
          err = OB_ENTRY_NOT_EXIST;
        }
      }
      while (OB_SUCCESS == err)
      {
        if (OB_SUCCESS != (err = log_reader.read_log(cmd, log_seq, log_data, data_len)))
        {
          if (OB_READ_NOTHING != err)
          {
            TBSYS_LOG(ERROR, "log_reader.read_log()=>%d", err);
          }
          else
          {
            err = OB_ENTRY_NOT_EXIST;
          }
        }
        else if ((int64_t)log_seq > log_id)
        {
          err = OB_ENTRY_NOT_EXIST;
        }
        else if ((int64_t)log_seq + 1 == log_id)
        {
          offset = log_reader.get_last_log_offset();
          break;
        }
        else if ((int64_t)log_seq == log_id) // 第一条日志
        {
          offset = 0;
          break;
        }
      }
      return err;
    }

    ObLogFileIdLocator::ObLogFileIdLocator(): max_log_file_id_getter_(NULL), first_log_id_getter_(NULL)
    {}

    ObLogFileIdLocator::~ObLogFileIdLocator()
    {}

    bool ObLogFileIdLocator::is_inited() const
    {
      return NULL != max_log_file_id_getter_ && NULL != first_log_id_getter_;
    }

    int ObLogFileIdLocator::init(IObMaxLogFileIdGetter* max_log_file_id_getter,
                                 IObFirstLogIdGetter* first_log_id_getter, const int64_t n_indexes)
    {
      int err = OB_SUCCESS;
      if (is_inited())
      {
        err = OB_INIT_TWICE;
      }
      else if (NULL == max_log_file_id_getter || NULL == first_log_id_getter || 0 >= n_indexes)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else if (OB_SUCCESS != (err = first_log_id_cache_.init(n_indexes)))
      {
        TBSYS_LOG(ERROR, "first_log_id_cache.init(n_indexes=%ld)=>%d", n_indexes, err);
      }
      else
      {
        max_log_file_id_getter_ = max_log_file_id_getter;
        first_log_id_getter_ = first_log_id_getter;
      }
      return err;
    }

    int ObLogFileIdLocator::get_file_id(int64_t log_id, int64_t& file_id)
    {
      int err = OB_SUCCESS;
      int64_t max_log_file_id = 0;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (OB_SUCCESS != (err = max_log_file_id_getter_->get_max_log_file_id(max_log_file_id)))
      {
        TBSYS_LOG(ERROR, "get_max_log_file_id()=>%d", err);
      }
      else if (OB_SUCCESS != (err = get_file_id_helper(max_log_file_id, log_id, file_id)))
      {
        TBSYS_LOG(ERROR, "get_file_id_helper(max_log_file_id=%ld, log_id=%ld)=>%d",
                  max_log_file_id, log_id, err);
      }
      return err;
    }

    int ObLogFileIdLocator::get_file_id_helper(const int64_t max_log_file_id, int64_t log_id, int64_t& file_id)
    {
      int err = OB_SUCCESS;
      int64_t first_log_id = 0;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      for(file_id = max_log_file_id; OB_SUCCESS == err && file_id > 0; file_id--)
      {
        if (OB_SUCCESS != (err = get_first_log_id(file_id, first_log_id)))
        {
          TBSYS_LOG(ERROR, "get_first_log_id(file_id=%ld)=>%d", file_id, err);
        }
        else if (log_id >= first_log_id)
        {
          break;
        }
      }
      if (OB_SUCCESS == err && file_id <= 0)
      {
        err = OB_ENTRY_NOT_EXIST;
      }
      return err;
    }

    int ObLogFileIdLocator::get_first_log_id(const int64_t file_id, int64_t& log_id)
    {
      int err = OB_SUCCESS;
      if (!is_inited())
      {
        err = OB_NOT_INIT;
      }
      else if (OB_SUCCESS != (err = first_log_id_cache_.get(file_id, log_id))
               && OB_ENTRY_NOT_EXIST != err)
      {
        TBSYS_LOG(ERROR, "index[%ld].copy()=>%d", file_id, err);
      }
      else if (OB_SUCCESS == err)
      {} // do nothing
      else if (OB_SUCCESS != (err = first_log_id_getter_->get_first_log_id(file_id, log_id))
               && OB_ENTRY_NOT_EXIST != err)
      {
        TBSYS_LOG(ERROR, "first_log_id_getter.get_first_log_id(file_id=%ld)=>%d", file_id, err);
      }
      else if (OB_SUCCESS == err &&
               OB_SUCCESS != (err = first_log_id_cache_.add(file_id, log_id)))
      {
        TBSYS_LOG(ERROR, "first_log_id_cache.add(file_id=%ld, log_id=%ld)=>%d", file_id, log_id);
      }
      TBSYS_LOG(INFO, "first_log_id(file_id=%ld, log_id=%ld)", file_id, log_id);
      return err;
    }

    ObFirstLogIdGetter::ObFirstLogIdGetter(): log_dir_(NULL)
    {}
    ObFirstLogIdGetter::~ObFirstLogIdGetter()
    {}

    int ObFirstLogIdGetter::init(const char* log_dir)
    {
      int err = OB_SUCCESS;
      if (NULL != log_dir_)
      {
        err = OB_INIT_TWICE;
      }
      else if (NULL == log_dir)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else
      {
        log_dir_ = log_dir;
      }
      return err;
    }

    int ObFirstLogIdGetter::get_first_log_id(const int64_t file_id, int64_t& log_id)
    {
      int err = OB_SUCCESS;
      if (NULL == log_dir_)
      {
        err = OB_NOT_INIT;
      }
      else if (OB_SUCCESS != (err = get_first_log_id_func(log_dir_, file_id, log_id)))
      {
        TBSYS_LOG(ERROR, "get_first_log_id_func(log_dir=%s, file_id=%ld)=>%d", log_dir_, file_id, err);
      }
      return err;
    }

    ObLogFileOffsetLocator::ObLogFileOffsetLocator() : log_dir_(NULL)
    {}

    ObLogFileOffsetLocator::~ObLogFileOffsetLocator()
    {}

    int ObLogFileOffsetLocator::init(const char* log_dir)
    {
      int err = OB_SUCCESS;
      if (NULL != log_dir_)
      {
        err = OB_INIT_TWICE;
      }
      else if (NULL == log_dir)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else
      {
        log_dir_ = log_dir;
      }
      return err;
    }

    int ObLogFileOffsetLocator::get_file_offset(const int64_t log_id, const int64_t file_id, int64_t& offset)
    {
      int err = OB_SUCCESS;
      if (NULL == log_dir_)
      {
        err = OB_NOT_INIT;
      }
      else if (OB_SUCCESS != (err = get_log_file_offset_func(log_dir_, file_id, log_id, offset)))
      {
        TBSYS_LOG(ERROR, "get_log_file_offset(log_dir=%s, file_id=%ld, log_id=%ld)=>%d", log_dir_, file_id, log_id, err);
      }
      return err;
    }

    int ObStoredMaxLogFileIdGetter:: get_max_log_file_id(int64_t& max_log_file_id)
    {
      int err = OB_SUCCESS;
      max_log_file_id = max_log_file_id_;
      return err;
    }

    int ObStoredMaxLogFileIdGetter:: set_max_log_file_id(const int64_t max_log_file_id)
    {
      int err = OB_SUCCESS;
      max_log_file_id_ = max_log_file_id;
      return err;
    }
  }; // end namespace updateserver
}; // end namespace oceanbase
