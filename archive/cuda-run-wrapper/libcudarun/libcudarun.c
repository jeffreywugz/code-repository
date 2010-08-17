/**
 * @file   libcudarun.c
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Sun Jul 19 21:12:18 2009
 * 
 * @brief  define a constructor cuda_set_best_device which will be called
 * before main and auto select a sequence of best devices to try for a cuda
 * program.
 *
 * @ingroup libcudarun
 * 
 */

#include <stdio.h>
#include "device_list.h"
#include "device_info_receiver.h"

int cuda_set_best_device() __attribute__ ((constructor));

/**
 * @brief get server addr
 * @ingroup libcudarun
 */

static const char* get_server_addr()
{
        return getenv("server_addr");
}

/**
 * @brief recv device info from daemon
 * @ingroup libcudarun
 * 
 */
static int device_list_recv_info(DeviceList* device_list)
{
        DeviceInfoReceiver device_info_receiver;
        UNClient client;
        const char* server_addr;
        int conn;

        server_addr = get_server_addr();
        unclient_init(&client, server_addr);
        conn = unclient_conn(&client);
        device_info_receiver_init(&device_info_receiver, conn,
                                  device_list->n_device);
        
        device_info_receiver_recv(&device_info_receiver);
        device_list_set_info(device_list, device_info_receiver.device_info);
        
        device_info_receiver_destroy(&device_info_receiver);
        unclient_destroy(&client);

        return 0;
}

/**
 * @brief this function will called before main
 *
 * @ingroup libcudarun
 */

int cuda_set_best_device()
{
        DeviceList device_list;
                
        device_list_init(&device_list);
        device_list_recv_info(&device_list);
        device_list_set_best(&device_list);
        device_list_destroy(&device_list);
        return 0;
}
