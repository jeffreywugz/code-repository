#ifndef _DEVICE_EVALUATOR_H_
#define _DEVICE_EVALUATOR_H_
/**
 * @file   device_evaluator.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Sun Jul 19 22:16:18 2009
 * 
 * @brief  define class DeviceEvaluator which can get a busy degree
 * from device state.
 *
 * @ingroup device
 */
 
#include "device.h"
/**
 * @brief evaluat a group of GPU devices
 *
 * @ingroup device
 */

struct DeviceEvaluator
{
        int n_device;
        Device* devices;
};
typedef struct DeviceEvaluator DeviceEvaluator;

/**
 * @memberof DeviceEvaluator
 * 
 */

int device_evaluator_init(DeviceEvaluator* device_evaluator, int n_device,
                          Device* device);

/**
 * @memberof DeviceEvaluator
 * 
 */

void device_evaluator_eval(DeviceEvaluator* device_evaluator);

#endif /* _DEVICE_EVALUATOR_H_ */
