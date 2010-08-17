#ifndef _DAEMON_CONN_H_
#define _DAEMON_CONN_H_
/**
 * @file   daemon_conn.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Mon Jul 20 09:29:06 2009
 * 
 * @brief  define class DaemonConn which will handle a single client connection.
 * 
 * @ingroup daemon
 * 
 */

#include "daemon.h"

/**
 * @brief handle a single client connection
 * @ingroup daemon
 */
struct DaemonConn
{
        Daemon* daemon;         /**< hold the essential info for connection */
        int conn;               /**< unix domain socket connection id */
};
typedef struct DaemonConn DaemonConn;

/**
 * @memberof DaemonConn
 * 
 */

DaemonConn* daemon_conn_new(Daemon* daemon, int conn);
/**
 * @memberof DaemonConn
 * 
 */

int daemon_conn_handle(DaemonConn* daemon_conn);
#endif /* _DAEMON_CONN_H_ */
