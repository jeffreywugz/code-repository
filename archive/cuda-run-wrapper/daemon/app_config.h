#ifndef _APP_OPTION_H_
#define _APP_OPTION_H_
/**
 * @file   app_config.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Sun Jul 19 21:00:09 2009
 * 
 * @brief  define class AppConfig which will parse cmd line and config file.
 *
 * @ingroup daemon
 * 
 */

#include "cmd_option.h"
#include "config_file.h"

/**
 * @brief class hold config var for daemon.
 * @ingroup daemon
 */

struct AppConfig
{
        CmdOption cmd_opt;      /**< cmd option parser proxy */
        ConfigFile cfg_file;    /**< config file parser proxy */
        int help;               /**< bool value represent whether user is ask for help*/
        int test;               /**< bool value represent whether program is in test mode*/
        int config_file;        /**< bool value represent whether user is specify config file name */
        char file[CONFIG_FILE_MAX_NAME_LEN]; /**< config file name */
        const char* server_addr; /**< server's unix domain socket path */
        int server_max_queued_requests; /**< how many requests the wait queue can hold */
        int daemon_max_cache_time_interval; /**< how long before the device info will become invalid */
        int daemon_max_cache_requests; /**< how many request will the daemon serve before the device info become invalid */
};
typedef struct AppConfig AppConfig;
/**
 * @memberof AppConfig
 * 
 */

int app_config_init(AppConfig* app_cfg, int argc, char** argv,
        const char* file);
/**
 * @memberof AppConfig
 * 
 */

void app_config_print(AppConfig* app_cfg);

#endif /* _APP_OPTION_H_ */
