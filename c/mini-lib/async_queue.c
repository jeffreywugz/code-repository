#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "util.h"
#include "async_queue.h"
#ifndef _ASYNC_QUEUE_H_
#define _ASYNC_QUEUE_H_
//<<<header

#include <pthread.h>
#include "queue.h"

struct AsyncQueue
{
        pthread_mutex_t mutex;
        pthread_cond_t cond;
        Queue queue;
        int n_waiting_threads;
};
typedef struct AsyncQueue AsyncQueue;
//header>>>
#endif /* _ASYNC_QUEUE_H_*/

//<<<func_list|get_func_list
int async_queue_init(AsyncQueue* async_queue, int capacity)
{
        int err;
        err = queue_init(&async_queue->queue, capacity);
        if(err < 0)panic("queue_init failed!\n");
        err = pthread_mutex_init(&async_queue->mutex, NULL);
        if(err != 0)sys_panic("pthread_mutex_init");
        err = pthread_cond_init(&async_queue->cond, NULL);
        if(err != 0)sys_panic("pthread_cond_init");
        async_queue->n_waiting_threads = 0;
        return 0;
}

int async_queue_push(AsyncQueue* async_queue, void* data)
{
        int err;
        pthread_mutex_lock(&async_queue->mutex);
        err = queue_push(&async_queue->queue, data);
        if(async_queue->n_waiting_threads > 0)
                pthread_cond_signal(&async_queue->cond);
        pthread_mutex_unlock(&async_queue->mutex);
        return err;
}

int async_queue_pop(AsyncQueue* async_queue, void** data)
{
        int err;
        pthread_mutex_lock(&async_queue->mutex);
        async_queue->n_waiting_threads++;
        while(queue_is_empty(&async_queue->queue))
                pthread_cond_wait(&async_queue->cond, &async_queue->mutex);
        err = queue_pop(&async_queue->queue, data);
        async_queue->n_waiting_threads--;
        pthread_mutex_unlock(&async_queue->mutex);
        return err;
}

int async_queue_destroy(AsyncQueue* async_queue)
{
        pthread_mutex_destroy(&async_queue->mutex);
        pthread_cond_destroy(&async_queue->cond);
        queue_destroy(&async_queue->queue);
        return 0;
}
//func_list>>>

//<<<test
#ifndef NOCHECK
#include "util.h"

static char output[4];

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

quick_define_tcase_reg(async_queue)
#endif /* NOCHECK */
//test>>>
