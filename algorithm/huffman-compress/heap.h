#ifndef  __HEAP_H__
#define  __HEAP_H__

#include "hfm_tree.h"

void create_heap(struct HfmTree* heap[], int len);
void adjust_down_heap(struct HfmTree* heap[], int len, int root);
void adjust_up_heap(struct HfmTree* heap[], int len, int child); 

#endif  /*__HEAP_H__*/
