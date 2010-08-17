#ifndef _THREAD_POOL_H_
#define _THREAD_POOL_H_
/**
 * @file   thread_pool.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Mon Jul 20 09:56:21 2009
 * 
 * @brief  define class ThreadPool.
 * 
 * @ingroup libphi
 * 
 */

#include <pthread.h>
#include "slice.h"
#include "async_queue.h"

#define THREAD_POOL_MAX_THREAD_NUM 128
struct ThreadPool
{
        int capacity;
        Slice task_allocator;
        pthread_t thread[THREAD_POOL_MAX_THREAD_NUM];
        AsyncQueue task_queue;
};
typedef struct ThreadPool ThreadPool;

int thread_pool_init(ThreadPool* thread_pool, int capacity);
int thread_pool_push(ThreadPool* thread_pool, void* (*func)(void*), void* arg);
int thread_pool_destroy(ThreadPool* thread_pool);
        
#endif /* _THREAD_POOL_H_ */
