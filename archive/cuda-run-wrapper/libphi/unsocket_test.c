#include <unistd.h>
#include <sys/types.h>
#include <check.h>
#include "util.h"
#include "unsocket.h"

#define MAX_MSG_LEN 256
START_TEST(unsocket_test)
{
        pid_t pid;
        const char* unix_path = "test.un";
        int max_queued_requests = 3;
        int err, len;
        int conn;
        char send_buf[MAX_MSG_LEN] = "hello...\n";
        char recv_buf[MAX_MSG_LEN];
        UNServer server;
        UNClient client;
        
        pid = fork();
        if(pid < 0)sys_panic("fork");
        if(pid >0){             /* client */
                sleep(1);
                err = unclient_init(&client, unix_path);
                fail_unless(err == 0, "client init failed");
                conn = unclient_conn(&client);
                fail_unless(conn > 0, "client connect failed");
                len = write(conn, send_buf, strlen(send_buf)+1);
                fail_unless(len == strlen(send_buf)+1, "client send failed");
                unclient_destroy(&client);
        } else {                /* server */
                err = unserver_init(&server, unix_path, max_queued_requests);
                fail_unless(err == 0, "server init failed");
                conn = unserver_accept(&server);
                fail_unless(conn > 0, "server accept failed");
                len = read(conn, recv_buf, sizeof(recv_buf));
                fail_unless(len == strlen(send_buf)+1, "server recv failed");
                fail_unless(strcmp(send_buf, recv_buf) == 0,
                            "send msg != recv msg");
                close(conn);
                unserver_destroy(&server);
        }
}END_TEST

void unsocket_test_add(Suite* s)
{
        TCase *tc_unsocket = tcase_create("unsocket");
        tcase_add_test(tc_unsocket, unsocket_test);
        suite_add_tcase(s, tc_unsocket);
}
