#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <sys/types.h>
#include <pthread.h>
#include "util.h"
#include "device_info_transfer.h"
#include "daemon.h"

struct DaemonConn
{
        Daemon* daemon;
        int conn;
};
typedef struct DaemonConn DaemonConn;

static DaemonConn* _daemon_conn_new()
{
        DaemonConn* daemon_conn;
        daemon_conn = malloc(sizeof(DaemonConn));
        if(daemon_conn == NULL)
                panic("no mem!\n");
        return daemon_conn;
}

static void _daemon_conn_init(DaemonConn* daemon_conn, Daemon* daemon, int conn)
{
        daemon_conn->daemon = daemon;
        daemon_conn->conn = conn;
}

static void _daemon_conn_destroy(DaemonConn* daemon_conn)
{
        free(daemon_conn);
}

DaemonConn* daemon_conn_new(Daemon* daemon, int conn)
{
        DaemonConn* daemon_conn;
        daemon_conn = _daemon_conn_new();
        _daemon_conn_init(daemon_conn, daemon, conn);
        return daemon_conn;
}

int daemon_conn_handle(DaemonConn* daemon_conn)
{
        DeviceInfoTransfer transfer;
        Daemon* daemon;
        int conn;
        int n_device;
        
        daemon = daemon_conn->daemon;
        conn = daemon_conn->conn;
        n_device = daemon->device_info_cache.device_list.n_device;
        device_info_transfer_init(&transfer, conn, n_device);
        daemon_info_get(daemon, transfer.device_info);
        device_info_transfer_send(&transfer);
        device_info_transfer_destroy(&transfer);
        close(conn);

        _daemon_conn_destroy(daemon_conn);
        return 0;
}
