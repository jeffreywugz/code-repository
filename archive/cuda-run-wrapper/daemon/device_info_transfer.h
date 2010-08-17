#ifndef _DEVICE_INFO_TRANSFER_H_
#define _DEVICE_INFO_TRANSFER_H_
/**
 * @file   device_info_transfer.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Mon Jul 20 09:30:57 2009
 * 
 * @brief  define class DeviceInfoTransfer which will send device info
 * to a connected client.
 *
 * @ingroup daemon
 * 
 */

#include "device_list.h"
#include "unsocket.h"

/**
 * @brief send device info to client
 *
 * @ingroup daemon
 */

struct DeviceInfoTransfer
{
        DeviceInfo device_info[MAX_DEVICE_COUNT];
        int n_device;
        int conn;
};
typedef struct DeviceInfoTransfer DeviceInfoTransfer;

/**
 * @memberof DeviceInfoTransfer
 * 
 */

int device_info_transfer_init(DeviceInfoTransfer* transfer,
                              int conn, int n_device);

/**
 * @memberof DeviceInfoTransfer
 * 
 */

int device_info_transfer_send(DeviceInfoTransfer* transfer);

/**
 * @memberof DeviceInfoTransfer
 * 
 */

int device_info_transfer_destroy(DeviceInfoTransfer* transfer);


#endif /* _DEVICE_INFO_TRANSFER_H_ */
