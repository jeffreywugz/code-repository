#ifndef _ASYNC_H_
#define _ASYNC_H_
#include <errno.h>
#include <signal.h>
#include <pthread.h>
#include "container.h"

struct sigaction* sigaction_set_handler(struct sigaction* action, void (*handler)(int))
{
        memset(action, 0, sizeof(*action));
        sigemptyset(&action->sa_mask);
        action->sa_handler = handler;
        return action;
}

struct sigaction* sigaction_with_handler(void (*handler)(int))
{
        static struct sigaction action;
        return sigaction_set_handler(&action, handler);
}

int pthread_create2(pthread_t* thread, const pthread_attr_t* attr,
                    handler_t handler, void* self, void* arg)
{
        return pthread_create(thread, attr, (function_t)closure_call_and_free,
                       closure_new(handler, self, arg));
}

typedef struct _async_queue_t {
        bool stop;
        pthread_mutex_t mutex;
        pthread_cond_t read_cond;
        pthread_cond_t write_cond;
        int n_read_waiting;
        int n_write_waiting;
        queue_t queue;
} async_queue_t;

int async_queue_init(async_queue_t* async_queue, void** buf, int capacity)
{
        int err = 0;
        pthread_mutexattr_t attr;
        err = queue_init(&async_queue->queue, buf, capacity);
        if(err)goto ret;
        err = pthread_mutexattr_init(&attr);
        if(err)goto ret;
        err = pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_NONE);
        if(err)goto mutexattr_fail;
        err = pthread_mutex_init(&async_queue->mutex, &attr);
        if(err)goto mutexattr_fail;
        err = pthread_cond_init(&async_queue->read_cond, NULL);
        if(err)goto read_cond_fail;
        err = pthread_cond_init(&async_queue->write_cond, NULL);
        if(err)goto write_cond_fail;
        async_queue->stop = false;
        async_queue->n_read_waiting = 0;
        async_queue->n_write_waiting = 0;
        goto ret;
write_cond_fail:
        pthread_cond_destroy(&async_queue->read_cond);
read_cond_fail:
        pthread_mutex_destroy(&async_queue->mutex);
mutexattr_fail:
        pthread_mutexattr_destroy(&attr);
ret:
        return err;
}

async_queue_t* async_queue_new(int capacity)
{
        void* mem = malloc(sizeof(async_queue_t) + sizeof(void*) * capacity);
        async_queue_t* queue;
        void** buf;
        if(!mem)return NULL;
        queue = (async_queue_t*)mem;
        buf = (void**)(queue+1);
        if(async_queue_init(queue, buf, capacity) != 0){
                free(mem);
                return NULL;
        }
        return queue;
}

void async_queue_report_stat(async_queue_t* async_queue, char* msg)
{
        return;
        fprintf(stderr, "%s: capatity: %d, queue items: %d, n_read: %d, n_write: %d\n",
                msg,
                async_queue->queue.capacity, queue_size(&async_queue->queue),
                async_queue->n_read_waiting,
                async_queue->n_write_waiting);
}

int async_queue_push(async_queue_t* async_queue, void* data)
{
        int err;
        if(async_queue->stop)return ECANCELED;
        async_queue_report_stat(async_queue, "push");
        err = pthread_mutex_lock(&async_queue->mutex);
        if(err)return err;
        async_queue->n_write_waiting++;
        while(!async_queue->stop && queue_is_full(&async_queue->queue))
                ckerr(pthread_cond_wait(&async_queue->write_cond, &async_queue->mutex));
        if(async_queue->stop){
                async_queue->n_write_waiting--;
                ckerr(pthread_mutex_unlock(&async_queue->mutex));
                return ECANCELED;
        }
        err = queue_push(&async_queue->queue, data);
        async_queue->n_write_waiting--;
        if(async_queue->n_read_waiting > 0)
                ckerr(pthread_cond_signal(&async_queue->read_cond));
        ckerr(pthread_mutex_unlock(&async_queue->mutex));
        return err;
}

int async_queue_pop(async_queue_t* async_queue, void** data)
{
        int err;
        if(async_queue->stop)return ECANCELED;
        async_queue_report_stat(async_queue, "pop");
        err = pthread_mutex_lock(&async_queue->mutex);
        async_queue->n_read_waiting++;
        while(!async_queue->stop && queue_is_empty(&async_queue->queue))
                ckerr(pthread_cond_wait(&async_queue->read_cond, &async_queue->mutex));
        if(async_queue->stop){
                async_queue->n_read_waiting--;
                ckerr(pthread_mutex_unlock(&async_queue->mutex));
                return ECANCELED;
        }
        err = queue_pop(&async_queue->queue, data);
        async_queue->n_read_waiting--;
        if(async_queue->n_write_waiting > 0)
                ckerr(pthread_cond_signal(&async_queue->write_cond));
        ckerr(pthread_mutex_unlock(&async_queue->mutex));
        return err;
}

void async_queue_wait(async_queue_t* async_queue)
{
        async_queue->stop = true;
        while(async_queue->n_write_waiting)
                pthread_cond_signal(&async_queue->write_cond);
        while(async_queue->n_read_waiting)
                pthread_cond_signal(&async_queue->read_cond);
}

int async_queue_destroy(async_queue_t* async_queue)
{
        if(!async_queue->stop)return -EBUSY;
        pthread_mutex_destroy(&async_queue->mutex);
        pthread_cond_destroy(&async_queue->read_cond);
        pthread_cond_destroy(&async_queue->write_cond);
        queue_destroy(&async_queue->queue);
        return 0;
}

int async_queue_del(async_queue_t* queue)
{
        int err = async_queue_destroy(queue);
        if(err)return err;
        free(queue);
        return err;
}

typedef struct _thread_pool_t {
        bool stop;
        int n_limit;
        int n_running;
        pthread_t* threads;
        handler_t handler;
        void* self;
        async_queue_t* queue;
} thread_pool_t;

void* thread_pool_handle(thread_pool_t* pool)
{
        while(!pool->stop){
                void* data = NULL;
                int err;
                if(pool->queue){
                        err = async_queue_pop(pool->queue, &data);
                        if(err)continue;
                }
                __sync_fetch_and_add(&pool->n_running, 1);
                pool->handler(pool->self, data);
                __sync_fetch_and_sub(&pool->n_running, 1);
        }
        return NULL;
}

int thread_pool_create(thread_pool_t* pool, int thread_limit, handler_t handler, void* self, async_queue_t* queue)
{
        pool->stop = false;
        pool->n_limit = thread_limit;
        pool->n_running = 0;
        pool->handler = handler;
        pool->self = self;
        pool->queue = queue;
        pool->threads = (pthread_t*)malloc(sizeof(pthread_t) * thread_limit);
        if(!pool->threads)return -ENOMEM;
        fprintf(stderr, "create thread...\n");
        for(int i = 0; i < thread_limit; i++){
                pthread_create(&pool->threads[i], NULL, (function_t)thread_pool_handle, pool);
        }
        return 0;
}

thread_pool_t* thread_pool_new(int thread_limit, handler_t handler, void* self, async_queue_t* queue)
{
        thread_pool_t* pool = (thread_pool_t*)malloc(sizeof(thread_pool_t));
        if(!pool)return NULL;
        if(thread_pool_create(pool, thread_limit, handler, self, queue) != 0)
                return NULL;
        return pool;
}

void thread_pool_wait(thread_pool_t* pool)
{
        pool->stop = true;
        if(pool->queue)async_queue_wait(pool->queue);
        for(int i = 0; i < pool->n_limit; i++)
                pthread_join(pool->threads[i], NULL);
}

int thread_pool_destroy(thread_pool_t* pool)
{
        if(!pool->stop)return -EBUSY;
        free(pool->threads);
        return 0;
}

int thread_pool_del(thread_pool_t* pool)
{
        int err = thread_pool_destroy(pool);
        if(err)return err;
        free(pool);
        return err;
}

thread_pool_t* workers_new(int thread_limit, handler_t handler, void* self, int qlen)
{
        async_queue_t* queue = NULL;
        if(qlen > 0){
                queue = async_queue_new(qlen);
                if(!queue)return NULL;
        }
        thread_pool_t* pool = thread_pool_new(thread_limit, handler, self, queue);
        if(!pool){
                async_queue_del(queue);
                return NULL;
        }
        return pool;
}

void workers_wait(thread_pool_t* pool)
{
        thread_pool_wait(pool);
}

int workers_del(thread_pool_t* pool)
{
        int err = 0;
        if(pool->queue){
                err = async_queue_del(pool->queue);
                if(err)return err;
        }
        err = thread_pool_del(pool);
        return err;
}

#ifdef TEST
#include <unistd.h>
static void* test_thread_producer(async_queue_t* async_queue, void* arg)
{
        //fprintf(stderr, "producer:\n");
        async_queue_push(async_queue, (void*)"hello!\n");
        usleep(100000);
        return NULL;
}

static void* test_thread_consumer(void* self, char* msg)
{
        cktrue(msg);
        fprintf(stderr, "%s", msg);
        usleep(100000);
        return NULL;
}

static void* test_thread_echo(void* self, char* msg)
{
        fprintf(stderr, "in thread\n");
        usleep(100000);
        return NULL;
}

long test_thread_pool()
{
        fprintf(stderr, "test echo:\n");
        thread_pool_t* echo = workers_new(10, (handler_t)test_thread_echo, NULL, 0);
        sleep(1);
        workers_wait(echo);
        workers_del(echo);
        
        fprintf(stderr, "test reader/writer:\n");
        thread_pool_t* readers = workers_new(10, (handler_t)test_thread_consumer, NULL, 10);
        thread_pool_t* writers = workers_new(10, (handler_t)test_thread_producer, readers->queue, 0);
        sleep(1);
        workers_wait(writers);
        workers_wait(readers);
        workers_del(readers);
        workers_del(writers);
        return 0;
}
#endif

#endif /* _ASYNC_H_ */
