#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <check.h>
#include "util.h"
#include "device_list.h"
#include "device_info_transfer.h"

int device_info_transfer_init(DeviceInfoTransfer* transfer,
                              int conn, int n_device)
{
        transfer->conn = conn;
        transfer->n_device = n_device;
        return 0;
}

int device_info_transfer_send(DeviceInfoTransfer* transfer)
{
        int len, write_len;
        len = transfer->n_device * sizeof(DeviceInfo);
        write_len = write(transfer->conn, transfer->device_info, len);
        assert(len == write_len);
        return 0;
}

int device_info_transfer_destroy(DeviceInfoTransfer* transfer)
{
        return 0;
}

