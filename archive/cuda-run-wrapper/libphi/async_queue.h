#ifndef _ASYNC_QUEUE_H_
#define _ASYNC_QUEUE_H_
/**
 * @file   async_queue.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Mon Jul 20 10:02:47 2009
 * 
 * @brief  define class AsyncQueue which is thread safe.
 * 
 * @ingroup libphi
 * 
 */

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
        
int async_queue_init(AsyncQueue* async_queue, int capacity);
int async_queue_push(AsyncQueue* async_queue, void* data);
int async_queue_pop(AsyncQueue* async_queue, void** data);
int async_queue_destroy(AsyncQueue* async_queue);
        
#endif /* _ASYNC_QUEUE_H_ */
