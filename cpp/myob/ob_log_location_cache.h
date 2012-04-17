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
#ifndef __OB_UPDATESERVER_OB_LOG_LOCATION_CACHE_H__
#define __OB_UPDATESERVER_OB_LOG_LOCATION_CACHE_H__
#include "ob_log_locator.h"

namespace oceanbase
{
  namespace updateserver
  {
    class ObLogLocationCache :public IObLogLocator
    {
      public:
        ObLogLocationCache(): location_cache_() {}
        virtual ~ObLogLocationCache() {}
        int init(int64_t cache_capacity);
        virtual int get_location(const int64_t log_id, ObLogLocation& location);
        virtual int enter_location(const ObLogLocation& location);
      private:
        ObRecentCache<int64_t, ObLogLocation> location_cache_;
    };
  }; // end namespace updateserver
}; // end namespace oceanbase
#endif /* __OB_UPDATESERVER_OB_LOG_LOCATION_CACHE_H__ */
