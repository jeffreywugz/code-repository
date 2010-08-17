#include "util.h"
#include "heap.h"

void create_heap(struct HfmTree* heap[], int len)
{
	int root;
	for(root=len/2; root; root--)
		adjust_down_heap(heap, len, root);
}

void adjust_down_heap(struct HfmTree* heap[], int len, int root)
{
	int child;
	for(child=2*root; child<=len; root=child, child=2*root){
		if(child<len && heap[child]->weight > heap[child+1]->weight)
			child++;
		if(heap[root]->weight < heap[child]->weight)break;
		swap(heap[root], heap[child]);
	}
}

void adjust_up_heap(struct HfmTree* heap[], int len, int child) 
{
	int root;
	for(root=child/2; root; child=root, root=child/2){
		if(heap[root]->weight <= heap[child]->weight)break;
		swap(heap[root], heap[child]);
	}
}
