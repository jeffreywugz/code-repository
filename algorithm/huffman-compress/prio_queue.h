#ifndef  __PRIO_QUEUE_H__
#define  __PRIO_QUEUE_H__

#include "heap.h"
#define  MAX_PRIO_QUEUE_LEN 1024

struct PrioQueue {
	struct HfmTree* heap[MAX_PRIO_QUEUE_LEN];
	int len;
};

void init_prio_queue(struct PrioQueue* prio_queue, struct HfmTree* tree[],
		int len);
void en_prio_queue(struct PrioQueue* prio_queue, struct HfmTree* element);
void de_prio_queue(struct PrioQueue* prio_queue, struct HfmTree** element);


#endif  /*__PRIO_QUEUE_H__*/
