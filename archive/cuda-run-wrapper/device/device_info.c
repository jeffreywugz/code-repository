#include <stdio.h>
#include "device_info.h"

void device_info_print(DeviceInfo* device_info)
{
        printf("free mem: %d, total mem: %d, delay: %d\n",
               device_info->free_mem, device_info->total_mem,
               device_info->delay);
}
