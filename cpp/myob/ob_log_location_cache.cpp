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
#include "ob_log_location_cache.h"

namespace oceanbase
{
  namespace updateserver
  {
    int ObLogLocationCache::init(int64_t cache_capacity)
    {
      int err = OB_SUCCESS;
      if (0 >= cache_capacity)
      {
        err = OB_INVALID_ARGUMENT;
      }
      else if (OB_SUCCESS != (err = location_cache_.init(cache_capacity)))
      {
        TBSYS_LOG(ERROR, "location_cache.init(cache_capacity=%ld)=>%d", cache_capacity, err);
      }
      return err;
    }

    int ObLogLocationCache::get_location(const int64_t log_id, ObLogLocation& location)
    {
      return location_cache_.get(log_id, location);
    }

    int ObLogLocationCache::enter_location(const ObLogLocation& location)
    {
      return location_cache_.add(location.log_id_, location);
    }
  }; // end namespace updateserver
}; // end namespace oceanbase
