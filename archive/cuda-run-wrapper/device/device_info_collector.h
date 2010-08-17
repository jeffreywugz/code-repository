#ifndef _DEVICE_INFO_COLLECTOR_H_
#define _DEVICE_INFO_COLLECTOR_H_
/**
 * @file   device_info_collector.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Sun Jul 19 22:17:28 2009
 * 
 * @brief  define class DeviceInfoCollector which will collect device info.
 *
 * @ingroup device
 * 
 */

#include <semaphore.h>
#include <cuda.h>
#include "device_info.h"

#define MAX_FILE_NAME_LEN 128
/**
 * @brief collect device info
 *
 * @ingroup device
 */

struct DeviceInfoCollector
{
        int id;
        CUcontext cu_ctx;
        sem_t* request_sem;
        sem_t* finish_sem;
        DeviceInfo device_info;
};
typedef struct DeviceInfoCollector DeviceInfoCollector;        

/**
 * @memberof DeviceInfoCollector
 * 
 */

int device_info_collector_init(DeviceInfoCollector* device_info_collector,
                               int device);
/**
 * @memberof DeviceInfoCollector
 * 
 */

int device_info_collector_collect(DeviceInfoCollector* device_info_collector);
        
/**
 * @memberof DeviceInfoCollector
 * 
 */

int device_info_collector_collect_async(
        DeviceInfoCollector* device_info_collector);
/**
 * @memberof DeviceInfoCollector
 * 
 */

int device_info_collector_wait(DeviceInfoCollector* device_info_collector);
/**
 * @memberof DeviceInfoCollector
 * 
 */

int device_info_collector_destroy(DeviceInfoCollector* device_info_collector);
#endif /* _DEVICE_INFO_COLLECTOR_H_ */
