#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include "unsocket.h"
#include "util.h"


static int unaddr_init(struct sockaddr_un* addr, const char* path)
{
        int len;
        addr->sun_family = AF_UNIX;
        strcpy(addr->sun_path, path);
        len = strlen(addr->sun_path) + sizeof(addr->sun_family);
        return len;
}

static int unserver_addr_init(UNServer* server)
{
        server->addr_len = unaddr_init(&server->addr, server->unix_path);
        return 0;
}

static int unserver_sock_init(UNServer* server)
{
        server->sock = socket(AF_UNIX, SOCK_STREAM, 0);
        if(server->sock < 0)sys_panic("socket");
        return 0;
}

static int unserver_init_(UNServer* server, const char* unix_path,
        int max_queued_requests)
{
        strcpy(server->unix_path, unix_path);
        server->max_queued_requests = max_queued_requests;
        return 0;
}

static int unserver_bind(UNServer* server)
{
        int err;
        unlink(server->unix_path);
        err = bind(server->sock, (struct sockaddr *)&server->addr,
                   server->addr_len);
        if(err < 0)sys_panic("bind");
        return 0;
}

static int unserver_listen(UNServer* server)
{
        int err;
        err = listen(server->sock, server->max_queued_requests);
        if(err < 0)sys_panic("listen");
        return 0;
}

int unserver_accept(UNServer* server)
{
        int conn;
        struct sockaddr_un client_addr;
        int client_addr_len;
        conn = accept(server->sock, (struct sockaddr*)&client_addr,
                      (socklen_t*)&client_addr_len);
        if(conn<0)sys_panic("accept");
        return conn;
}

int unserver_init(UNServer* server, const char* unix_path,
                  int max_queued_requests)
{
        unserver_init_(server, unix_path, max_queued_requests);
        unserver_sock_init(server);
        unserver_addr_init(server);
        unserver_bind(server);
        unserver_listen(server);
        return 0;
}

int unserver_destroy(UNServer* server)
{
        close(server->sock);
        return 0;
}


static int unclient_addr_init(UNClient* client)
{
        client->server_addr_len = unaddr_init(&client->server_addr,
                                       client->server_unix_path);
        return 0;
}

static int unclient_sock_init(UNClient* client)
{
        client->sock = socket(AF_UNIX, SOCK_STREAM, 0);
        if(client->sock < 0)sys_panic("socket");
        return 0;
}

static int unclient_init_(UNClient* client, const char* server_unix_path)
{
        strcpy(client->server_unix_path, server_unix_path);
        return 0;
}

int unclient_init(UNClient* client, const char* server_unix_path)
{
        unclient_init_(client, server_unix_path);
        unclient_sock_init(client);
        unclient_addr_init(client);
        return 0;
}

int unclient_conn(UNClient* client)
{
        int err;
        err = connect(client->sock, (struct sockaddr*)&client->server_addr,
                      client->server_addr_len);
        if(err < 0)sys_panic("connect");
        return client->sock;
}

int unclient_destroy(UNClient* client)
{
        close(client->sock);
        return 0;
}
