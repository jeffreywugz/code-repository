#include <stdio.h>
#include "device.h"

int device_init(Device* device, int id)
{
        device->id = id;
        return 0;
}

void device_print(Device* device)
{
        printf("id: %d, score: %f\n", device->id, device->score);
        device_info_print(&device->info);
}
