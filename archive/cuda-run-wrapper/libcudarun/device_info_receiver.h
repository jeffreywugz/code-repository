#ifndef _DEVICE_INFO_RECEIVER_H_
#define _DEVICE_INFO_RECEIVER_H_
/**
 * @file   device_info_receiver.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Sun Jul 19 21:15:06 2009
 * 
 * @brief  define class DeviceInfoReceiver which will recv device from a
 * connected daemon.
 *
 * @ingroup libcudarun
 * 
 */

#include "unsocket.h"

/**
 * @brief receive device info from daemon
 * @ingroup libcudarun
 */

struct DeviceInfoReceiver
{
        DeviceInfo device_info[MAX_DEVICE_COUNT];
        int n_device;
        int conn;
};
typedef struct DeviceInfoReceiver DeviceInfoReceiver;

/**
 * @memberof DeviceInfoReceiver
 * 
 */

int device_info_receiver_init(DeviceInfoReceiver* receiver, int conn,
                              int n_device);

/**
 * @memberof DeviceInfoReceiver
 * 
 */

int device_info_receiver_recv(DeviceInfoReceiver* receiver);

/**
 * @memberof DeviceInfoReceiver
 * 
 */

int device_info_receiver_destroy(DeviceInfoReceiver* receiver);


#endif /* _DEVICE_INFO_RECEIVER_H_ */
