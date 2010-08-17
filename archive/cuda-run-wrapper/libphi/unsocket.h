#ifndef _UNSOCKET_H_
#define _UNSOCKET_H_
/**
 * @file   unsocket.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Mon Jul 20 10:06:22 2009
 * 
 * @brief  define class UNServer and UNClient which communicate by unix domain
 * socket.
 * 
 * @ingroup libphi
 * 
 */

#include <sys/socket.h>
#include <sys/un.h>

#define MAX_NAME_LEN 256

struct UNServer
{
        char unix_path[MAX_NAME_LEN];
        int sock;
        struct sockaddr_un addr;
        int addr_len;
        int max_queued_requests;
};
typedef struct UNServer UNServer;
int unserver_init(UNServer* server, const char* unix_path,
                  int max_queued_requests);
int unserver_destroy(UNServer* server);
int unserver_accept(UNServer* server);

struct UNClient
{
        char server_unix_path[MAX_NAME_LEN];
        int sock;
        struct sockaddr_un server_addr;
        int server_addr_len;
};
typedef struct UNClient UNClient;
int unclient_init(UNClient* client, const char* server_unix_path);
int unclient_destroy(UNClient* client);
int unclient_conn(UNClient* client);

#endif /* _UNSOCKET_H_ */
