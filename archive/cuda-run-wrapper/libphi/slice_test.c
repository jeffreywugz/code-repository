#include <check.h>
#include "slice.h"

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

void slice_test_add(Suite* s)
{
        TCase *tc_slice = tcase_create("slice");
        tcase_add_test(tc_slice, slice_test);
        suite_add_tcase(s, tc_slice);
}




