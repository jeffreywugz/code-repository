#include <assert.h>
#include "stack.h"
#ifndef _STACK_H_
#define _STACK_H_
//<<<header


#define MAX_STACK_LEN 256
struct Stack
{
        int capacity;
        int top;
        void* stack[MAX_STACK_LEN];
};
typedef struct Stack Stack;



//header>>>
#endif /* _STACK_H_*/

//<<<func_list|get_func_list


int stack_init(Stack* stack, int capacity)
{
        assert(capacity <= MAX_STACK_LEN);
        stack->capacity = capacity;
        stack->top = 0;
        return 0;
}

int stack_size(Stack* stack)
{
        return stack->top;
}

int stack_is_empty(Stack* stack)
{
        return stack_size(stack) == 0;
}

int stack_is_full(Stack* stack)
{
        return stack_size(stack) == stack->capacity;
}

int stack_push(Stack* stack, void* data)
{
        if(stack_is_full(stack))
                return -1;
        stack->stack[stack->top++] = data;
        return 0;
}

int stack_pop(Stack* stack, void** data)
{
        if(stack_is_empty(stack))
                return -1;
        *data = stack->stack[--stack->top];
        return 0;
}

int stack_destroy(Stack* stack)
{
        return 0;
}
//func_list>>>

//<<<test
#ifndef NOCHECK
#include "util.h"

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

quick_define_tcase_reg(stack)


#endif /* NOCHECK */
//test>>>
