/**
 * @file   main.c
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Sun Jul 19 20:57:54 2009
 * 
 * @brief  define main function.
 * @ingroup daemon
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "app_config.h"
#include "daemon.h"
#include "all_test.h"

/**
 * @brief actual main of daemon
 * @ingroup daemon
 */

static int app_main(const char* unix_path, int max_queued_requests,
                    int max_cache_time_interval, int max_cache_requests)
{
        Daemon daemon;
        daemon_init(&daemon, unix_path, max_queued_requests,
                    max_cache_time_interval, max_cache_requests);
        daemon_run(&daemon);
        daemon_destroy(&daemon);
        return 0;
}

/** 
 * @brief main function of daemon
 * @ingroup daemon
 */
int main(int argc, char *argv[])
{
         AppConfig app_cfg;
         const char* file = "config.conf";
         int err;

         if(app_config_init(&app_cfg, argc, argv, file)<0)
                 panic("config init error!\n");
         if(app_cfg.test)
                 return all_test();

         err = app_main(app_cfg.server_addr,
                        app_cfg.server_max_queued_requests,
                        app_cfg.daemon_max_cache_time_interval,
                        app_cfg.daemon_max_cache_requests);
        
        return err;
}
