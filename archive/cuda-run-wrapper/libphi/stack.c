#include <assert.h>
#include "stack.h"

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
