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
#include "device_info_receiver.h"

int device_info_receiver_init(DeviceInfoReceiver* receiver, int conn,
                              int n_device)
{
        receiver->conn = conn;
        receiver->n_device = n_device;
        return 0;
}

int device_info_receiver_recv(DeviceInfoReceiver* receiver)
{
        int len, read_len;
        len = receiver->n_device * sizeof(DeviceInfo);
        read_len = read(receiver->conn, receiver->device_info, len);
        assert(len == read_len);
        return 0;
}

int device_info_receiver_destroy(DeviceInfoReceiver* receiver)
{
        return 0;
}

