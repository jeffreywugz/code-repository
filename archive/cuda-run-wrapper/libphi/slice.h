#ifndef _SLICE_H_
#define _SLICE_H_
/**
 * @file   slice.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Mon Jul 20 10:01:47 2009
 * 
 * @brief  define class Slice which manage equal sized memory chunk.
 * 
 * @ingroup libphi
 * 
 */

#include "stack.h"

struct Slice
{
        int capacity;
        int size;
        char* buf;
        Stack free_chunk;
        
};
typedef struct Slice Slice;

int slice_init(Slice* slice, int capacity, int size);
void* slice_get(Slice* slice);
int slice_put(Slice* slice, void* p);
int slice_destroy(Slice* slice);
        
#endif /* _SLICE_H_ */
