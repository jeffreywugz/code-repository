#ifndef _DEVICE_INFO_H_
#define _DEVICE_INFO_H_
/**
 * @file   device_info.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Sun Jul 19 22:10:32 2009
 * 
 * @brief  define class DeviceInfo which represent device info.
 *
 * @ingroup device
 * 
 */

/**
 * @brief represent a single device info
 * @ingroup device
 * 
 */

struct DeviceInfo
{
        unsigned free_mem, total_mem;
        unsigned delay;
};
typedef struct DeviceInfo DeviceInfo;

/**
 * @memberof DeviceInfo
 * 
 */

void device_info_print(DeviceInfo* device_info);

#endif /* _DEVICE_INFO_H_ */
