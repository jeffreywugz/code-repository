#ifndef _DEVICE_INFO_CACHE_H_
#define _DEVICE_INFO_CACHE_H_
/**
 * @file   device_info_cache.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Mon Jul 20 09:29:41 2009
 * 
 * @brief  define class DeviceInfoCache which will be a proxy to get device
 * info.
 * 
 * @ingroup daemon
 * 
 */

#include <pthread.h>
#include "device_list.h"

/**
 * @brief get device info and cache it
 *
 * @ingroup daemon
 * 
 */

struct DeviceInfoCache
{
        pthread_mutex_t device_info_lock; /**< mutex lock for device info */
        pthread_t update_thread; /**< the thread to update device info async */
        DeviceList device_list; /**< the group of GPU devices */
        DeviceInfo device_info[MAX_DEVICE_COUNT]; /**< cached device info */
        long long time_stamp;   /**< current cached device info born time */
        long long max_cache_time_interval; /**< how long can cached info survive */
        int max_cache_requests; /**< how many requests we can serve use cached info */
        int cache_requests;     /**< how many requests we served use current cached info */
};
typedef struct DeviceInfoCache DeviceInfoCache;

/**
 * @memberof DeviceInfoCache
 * 
 */

int device_info_cache_init(DeviceInfoCache* device_info_cache,
        long long max_cache_time_interval, int max_cache_requests);
/**
 * @memberof DeviceInfoCache
 * 
 */

int device_info_cache_get(DeviceInfoCache* device_info_cache,
                          DeviceInfo* device_info);
/**
 * @memberof DeviceInfoCache
 * 
 */

int device_info_cache_destroy(DeviceInfoCache* device_info_cache);
#endif /* _DEVICE_INFO_CACHE_H_ */
