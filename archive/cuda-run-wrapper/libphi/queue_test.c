#include <check.h>
#include "queue.h"

START_TEST(queue_test)
{
        Queue queue;
        char* s1 = "just ";
        char* s2 = "for ";
        char* s3 = "fun!";
        char* s;
        char output[128] = "";
        queue_init(&queue, 20);
        queue_push(&queue, s1);
        queue_push(&queue, s2);
        queue_push(&queue, s3);
        while(!queue_is_empty(&queue)){
                queue_pop(&queue, (void**)&s);
                strcat(output, s);
        }
        fail_unless(strcmp(output, "just for fun!") == 0, "queue test");
}END_TEST

void queue_test_add(Suite* s)
{
        TCase *tc_queue = tcase_create("queue");
        tcase_add_test(tc_queue, queue_test);
        suite_add_tcase(s, tc_queue);
}
