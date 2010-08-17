#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "util.h"
#include "async_queue.h"

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
