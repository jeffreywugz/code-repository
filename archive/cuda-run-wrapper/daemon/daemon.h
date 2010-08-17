#ifndef _DAEMON_H_
#define _DAEMON_H_
/**
 * @file   daemon.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Mon Jul 20 09:28:27 2009
 * 
 * @brief  define class Daemon which will handle all client connections.
 * 
 * @ingroup daemon
 * 
 */

#include "device_info_cache.h"
#include "unsocket.h"
#include "thread_pool.h"

/**
 * @brief handle all client connection
 * @ingroup daemon
 */

struct Daemon
{
        DeviceInfoCache device_info_cache; /**< get device info */
        ThreadPool thread_pool; /**< launch thread to handle connection */
        UNServer server;        /**< commuction through unix domain socket */
};
typedef struct Daemon Daemon;

/**
 * @memberof Daemon
 * 
 */
int daemon_init(Daemon* daemon, const char* unix_path,
                int max_queued_requests, long long max_cache_time_interval,
                int max_cache_requests);
/**
 * @memberof Daemon
 * 
 */

int daemon_info_get(Daemon* daemon, DeviceInfo* device_info);
/**
 * @memberof Daemon
 * 
 */

int daemon_run(Daemon* daemon);
/**
 * @memberof Daemon
 * 
 */

int daemon_destroy(Daemon* daemon);

#endif /* _DAEMON_H_ */
