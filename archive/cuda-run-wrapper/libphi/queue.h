#ifndef _QUEUE_H_
#define _QUEUE_H_
/**
 * @file   queue.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Mon Jul 20 10:07:43 2009
 * 
 * @brief  define class Queue.
 * 
 * @ingroup libphi
 * 
 */

#define MAX_QUEUE_LEN 256
struct Queue
{
        int capacity;
        int front, rear;
        void* queue[MAX_QUEUE_LEN];
};
typedef struct Queue Queue;

int queue_init(Queue* queue, int capacity);
int queue_destroy(Queue* queue);
int queue_size(Queue* queue);
int queue_is_empty(Queue* queue);
int queue_is_full(Queue* queue);
int queue_push(Queue* queue, void* data);
int queue_pop(Queue* queue, void** data);
        
#endif /* _QUEUE_H_ */
#define MAX_QUEUE_LEN 256
