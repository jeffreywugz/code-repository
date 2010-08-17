#include <stdlib.h>
#include "slice.h"
#ifndef _SLICE_H_
#define _SLICE_H_
//<<<header

#include "stack.h"

struct Slice
{
        int capacity;
        int size;
        char* buf;
        Stack free_chunk;
        
};
typedef struct Slice Slice;



//header>>>
#endif /* _SLICE_H_*/

//<<<func_list|get_func_list


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
//func_list>>>

//<<<test
#ifndef NOCHECK
#include "util.h"

START_TEST(slice_test)
{
        Slice slice;
        int capacity = 20;
        int size = sizeof(int);
        int *p;
        int err;
        slice_init(&slice, capacity, size);
        p = slice_get(&slice);
        fail_unless(p != NULL, "slice_get");
        *p =2;
        err = slice_put(&slice, p);
        fail_unless(err == 0, "slice_put");
        slice_destroy(&slice);
}END_TEST

quick_define_tcase_reg(slice)

#endif /* NOCHECK */
//test>>>
