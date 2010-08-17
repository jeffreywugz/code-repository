#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include "thread_pool.h"

struct ThreadTask
{
        void* (*func)(void*);
        void* arg;
};
typedef struct ThreadTask ThreadTask;

static int thread_task_init(ThreadTask* thread_task,
                            void* (*func)(void*), void* arg)
{
        thread_task->func = func;
        thread_task->arg = arg;
        return 0;
}

static void* thread_task_run(ThreadTask* thread_task)
{
        return thread_task->func(thread_task->arg);
}

static int thread_pool_task_handle(ThreadPool* thread_pool)
{
        ThreadTask* thread_task;
        int err;
        while(1){
                err = async_queue_pop(&thread_pool->task_queue,
                                      (void**)&thread_task);
                assert(err == 0);
                thread_task_run(thread_task);
                slice_put(&thread_pool->task_allocator, thread_task);
        }
        return 0;
}

int thread_pool_init(ThreadPool* thread_pool, int capacity)
{
        int err;
        int i;
        assert(capacity < THREAD_POOL_MAX_THREAD_NUM);
        thread_pool->capacity = capacity;
        err = slice_init(&thread_pool->task_allocator,
                         capacity, sizeof(ThreadTask));
        if(err < 0)return err;
        err = async_queue_init(&thread_pool->task_queue, capacity);
        if(err < 0)return err;
        for(i=0; i<capacity; i++){
                err = pthread_create(thread_pool->thread + i, NULL,
                                     (void*)thread_pool_task_handle,
                                     (void*)thread_pool);
                if(err != 0)return -1;
        }
        return err;
}

int thread_pool_push(ThreadPool* thread_pool, void* (*func)(void*), void* arg)
{
        int err;
        ThreadTask* thread_task;
        thread_task = slice_get(&thread_pool->task_allocator);
        thread_task_init(thread_task, func, arg);
        if(thread_task == NULL)return -1;
        err = async_queue_push(&thread_pool->task_queue, thread_task);
        return err;
}

int thread_pool_destroy(ThreadPool* thread_pool)
{
        slice_destroy(&thread_pool->task_allocator);
        async_queue_destroy(&thread_pool->task_queue);
        return 0;
}
