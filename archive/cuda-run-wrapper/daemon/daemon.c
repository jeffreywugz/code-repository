#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <sys/types.h>
#include <pthread.h>
#include "util.h"
#include "device_info_transfer.h"
#include "daemon.h"
#include "daemon_conn.h"

#define MAX_MSG_LEN 256


int daemon_init(Daemon* daemon, const char* unix_path,
                int max_queued_requests, long long max_cache_time_interval,
                int max_cache_requests)
{
        device_info_cache_init(&daemon->device_info_cache,
                max_cache_time_interval, max_cache_requests);
        thread_pool_init(&daemon->thread_pool, max_queued_requests);
        unserver_init(&daemon->server, unix_path, max_queued_requests);
        return 0;
}

static int daemon_handle(Daemon* daemon, int conn)
{
        DaemonConn* daemon_conn;
        int err;

        daemon_conn = daemon_conn_new(daemon, conn);
        err = thread_pool_push(&daemon->thread_pool,
                               (void*)daemon_conn_handle, daemon_conn);
        assert(err == 0);
        
        return 0;
}

int daemon_info_get(Daemon* daemon, DeviceInfo* device_info)
{
        return device_info_cache_get(&daemon->device_info_cache, device_info);
}

int daemon_run(Daemon* daemon)
{
        int conn;
        while(1){
                debug("unserver_accept\n");
                conn = unserver_accept(&daemon->server);
                if(conn < 0)panic("daemon accept connection error!\n");
                daemon_handle(daemon, conn);
        }
        return 0;
}

int daemon_destroy(Daemon* daemon)
{
        device_info_cache_destroy(&daemon->device_info_cache);
        thread_pool_destroy(&daemon->thread_pool);
        unserver_destroy(&daemon->server);
        return 0;
}

