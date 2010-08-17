#ifndef _DEVICE_LIST_H_
#define _DEVICE_LIST_H_
/**
 * @file   device_list.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Sun Jul 19 22:08:58 2009
 * 
 * @brief  define class DeviceList which abstract the group of GPU devices.
 *
 * @ingroup device
 * 
 */

#include "device.h"
#include "device_info_collector.h"

#define MAX_DEVICE_COUNT 4
/**
 * @brief represent a group of GPU device
 *
 * @ingroup device
 */

struct DeviceList
{
        int n_device;
        Device devices[MAX_DEVICE_COUNT];
#ifdef DAEMON
        DeviceInfoCollector device_info_collector[MAX_DEVICE_COUNT];
#endif
};
typedef struct DeviceList DeviceList;

/**
 * @memberof DeviceList
 * 
 */

int device_list_init(DeviceList* device_list);
/**
 * @memberof DeviceList
 * 
 */

void device_list_set_best(DeviceList* device_list);
#ifdef DAEMON
/**
 * @memberof DeviceList
 * 
 */

int device_list_get_info(DeviceList* device_list, DeviceInfo* device_info);
#endif
/**
 * @memberof DeviceList
 * 
 */

void device_list_set_info(DeviceList* device_list, DeviceInfo* device_info);
/**
 * @memberof DeviceList
 * 
 */

void device_list_print(DeviceList* device_list);
/**
 * @memberof DeviceList
 * 
 */

int device_list_destroy(DeviceList* device_list);

#endif /* _DEVICE_LIST_H_ */
