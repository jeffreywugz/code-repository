#include <assert.h>
#include "queue.h"
#ifndef _QUEUE_H_
#define _QUEUE_H_
//<<<header
#define MAX_QUEUE_LEN 256

struct Queue
{
        int capacity;
        int front, rear;
        void* queue[MAX_QUEUE_LEN];
};
typedef struct Queue Queue;
//header>>>
#endif /* _QUEUE_H_*/

//<<<func_list|get_func_list
int queue_init(Queue* queue, int capacity)
{
        assert(capacity < MAX_QUEUE_LEN && capacity > 0);
        queue->capacity = capacity + 1;
        queue->front = 0;
        queue->rear = 0;
        return 0;
}

int queue_destroy(Queue* queue)
{
        return 0;
}

int queue_size(Queue* queue)
{
        return (queue->rear + queue->capacity - queue->front) % queue->capacity;
}

int queue_is_empty(Queue* queue)
{
        return queue_size(queue) == 0;
}

int queue_is_full(Queue* queue)
{
        return queue_size(queue) == queue->capacity-1;
}

int queue_push(Queue* queue, void* data)
{
        if(queue_is_full(queue))
                return -1;
        queue->queue[queue->rear] = data;
        queue->rear = (queue->rear + 1) % queue->capacity;
        return 0;
}

int queue_pop(Queue* queue, void** data)
{
        if(queue_is_empty(queue))
                return -1;
        *data = queue->queue[queue->front];
        queue->front = (queue->front + 1) % queue->capacity;
        return 0;
}

//func_list>>>

//<<<test
#ifndef NOCHECK
#include "util.h"

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

quick_define_tcase_reg(queue)
#endif /* NOCHECK */
//test>>>
