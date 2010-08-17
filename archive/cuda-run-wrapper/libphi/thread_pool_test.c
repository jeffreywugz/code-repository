#include <unistd.h>
#include <pthread.h>
#include <check.h>
#include "async_queue.h"
#include "thread_pool.h"

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

START_TEST(thread_pool_test)
{
        AsyncQueue async_queue;
        ThreadPool thread_pool;
        async_queue_init(&async_queue, 20);
        thread_pool_init(&thread_pool, 20);
        thread_pool_push(&thread_pool, (void*)consumer, &async_queue);
        thread_pool_push(&thread_pool, (void*)consumer, &async_queue);
        thread_pool_push(&thread_pool, (void*)producer, &async_queue);
        thread_pool_push(&thread_pool, (void*)consumer, &async_queue);
        output[3] = '\0';
        sleep(1);
        fail_unless(strcmp(output, "aaa")==0, "thread_pool");
}END_TEST

void thread_pool_test_add(Suite* s)
{
        TCase *tc_thread_pool = tcase_create("thread_pool");
        tcase_add_test(tc_thread_pool, thread_pool_test);
        suite_add_tcase(s, tc_thread_pool);
}
