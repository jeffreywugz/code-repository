#include <string.h>
#include "prio_queue.h"

void init_prio_queue(struct PrioQueue* prio_queue, struct HfmTree* tree[],
		int len)
{
	memcpy(prio_queue->heap+1, tree, sizeof(struct HfmTree*)*len);
	prio_queue->len=len;
	create_heap(prio_queue->heap, prio_queue->len);
}

void en_prio_queue(struct PrioQueue* prio_queue, struct HfmTree* element)
{
	prio_queue->heap[++prio_queue->len]=element;
	adjust_up_heap(prio_queue->heap, prio_queue->len, prio_queue->len);
}

void de_prio_queue(struct PrioQueue* prio_queue, struct HfmTree** element)
{
	*element=prio_queue->heap[1];
	prio_queue->heap[1]=prio_queue->heap[prio_queue->len--];
	adjust_down_heap(prio_queue->heap, prio_queue->len, 1);
}
