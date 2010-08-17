#ifndef _DEVICE_H_
#define _DEVICE_H_
/**
 * @file   device.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Sun Jul 19 22:12:09 2009
 * 
 * @brief  define class Device which abstract a single GPU device.
 * 
 * @ingroup device
 * 
 */

#include "device_info.h"

/**
 * @brief used for evaluating a device
 * @ingroup device
 */

struct Device
{
        int id;
        double score;
        DeviceInfo info;
};
typedef struct Device Device;

/**
 * @memberof Device
 * 
 */

int device_init(Device* device, int id);
/**
 * @memberof Device
 * 
 */

void device_print(Device* device);

#endif /* _DEVICE_H_ */
