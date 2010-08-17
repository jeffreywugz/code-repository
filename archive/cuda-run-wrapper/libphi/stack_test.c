#include <check.h>
#include "stack.h"

START_TEST(stack_test)
{
        Stack stack;
        char* s1 = "just ";
        char* s2 = "for ";
        char* s3 = "fun!";
        char* s;
        char output[128] = "";
        stack_init(&stack, 20);
        stack_push(&stack, s3);
        stack_push(&stack, s2);
        stack_push(&stack, s1);
        while(!stack_is_empty(&stack)){
                stack_pop(&stack, (void**)&s);
                strcat(output, s);
        }
        fail_unless(strcmp(output, "just for fun!") == 0, "stack test");
}END_TEST

void stack_test_add(Suite* s)
{
        TCase *tc_stack = tcase_create("stack");
        tcase_add_test(tc_stack, stack_test);
        suite_add_tcase(s, tc_stack);
}


