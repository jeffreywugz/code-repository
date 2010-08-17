#include <stdlib.h>
#include "slice.h"

int slice_init(Slice* slice, int capacity, int size)
{
        int err, i;
        slice->capacity = capacity;
        slice->size = size;
        slice->buf = malloc(capacity*size);
        if(slice->buf == NULL)return -1;
        err = stack_init(&slice->free_chunk, capacity);
        if(err < 0)return err;
        for(i=0; i<capacity; i++)
                stack_push(&slice->free_chunk, slice->buf + i*size);
        return 0;
}

void* slice_get(Slice* slice)
{
        void* p;
        int err;
        err = stack_pop(&slice->free_chunk, &p);
        if(err < 0)return NULL;
        return p;
}

int slice_put(Slice* slice, void* p)
{
        int err;
        err = stack_push(&slice->free_chunk, p);
        if(err < 0)return -1;
        return 0;
}

int slice_destroy(Slice* slice)
{
        stack_destroy(&slice->free_chunk);
        free(slice->buf);
        return 0;
}
