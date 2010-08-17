#include <pthread.h>
#include <check.h>
#include "async_queue.h"

char output[4];

static void* producer(AsyncQueue* async_queue)
{
        async_queue_push(async_queue, output + 0);
        async_queue_push(async_queue, output + 1);
        async_queue_push(async_queue, output + 2);
        return NULL;
}

static void* consumer(AsyncQueue* async_queue)
{
        int err;
        char* c;
        err = async_queue_pop(async_queue, (void**)&c);
        fail_unless(err == 0);
        *c = 'a';
        return NULL;
}

START_TEST(async_queue_test)
{
        AsyncQueue async_queue;
        pthread_t thread1, thread2, thread3, thread4;
        async_queue_init(&async_queue, 20);
        pthread_create(&thread1, NULL, (void*)consumer, (void*)&async_queue);
        pthread_create(&thread2, NULL, (void*)consumer, (void*)&async_queue);
        pthread_create(&thread3, NULL, (void*)producer, (void*)&async_queue);
        pthread_create(&thread4, NULL, (void*)consumer, (void*)&async_queue);
        pthread_join(thread1, NULL);
        pthread_join(thread2, NULL);
        pthread_join(thread3, NULL);
        pthread_join(thread4, NULL);
        output[3] = '\0';
        fail_unless(strcmp(output, "aaa")==0, "async-queue");
}END_TEST

void async_queue_test_add(Suite* s)
{
        TCase *tc_queue = tcase_create("async-queue");
        tcase_add_test(tc_queue, async_queue_test);
        suite_add_tcase(s, tc_queue);
}
