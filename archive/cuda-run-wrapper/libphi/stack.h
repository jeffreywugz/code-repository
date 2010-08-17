#ifndef _STACK_H_
#define _STACK_H_
/**
 * @file   stack.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Mon Jul 20 10:08:14 2009
 * 
 * @brief  define class Stack.
 * 
 * @ingroup libphi
 * 
 */

#define MAX_STACK_LEN 256
struct Stack
{
        int capacity;
        int top;
        void* stack[MAX_STACK_LEN];
};
typedef struct Stack Stack;

int stack_init(Stack* stack, int capacity);
int stack_size(Stack* stack);
int stack_is_empty(Stack* stack);
int stack_is_full(Stack* stack);
int stack_push(Stack* stack, void* data);
int stack_pop(Stack* stack, void** data);
int stack_destroy(Stack* stack);
        
#endif /* _STACK_H_ */
