#include <assert.h>
#include "queue.h"

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

