#ifndef _CALL_H_
#define _CALL_H_
#include "core.h"
#include "container.h"
#include "strlib.h"

typedef void* (*function_t)(void* arg);
typedef void* (*handler_t)(void* self, void* arg);
typedef void* (*iter_t)(void* self);

typedef struct _closure_t {
        handler_t func;
        void* self;
        void* arg;
} closure_t;

closure_t* closure_init(closure_t* closure, handler_t func, void * self, void* arg)
{
        *closure = (closure_t){func, self, arg};
        return closure;
}

closure_t* closure_new(handler_t func, void * self, void* arg)
{
        closure_t* closure = (closure_t*)malloc(sizeof(closure_t));
        if(!closure)return NULL;
        return closure_init(closure, func, self, arg);
}

void* closure_call(closure_t* closure)
{
        return closure->func(closure->self, closure->arg);
}

void* closure_call_and_free(closure_t* closure)
{
        void* ret = closure_call(closure);
        free(closure);
        return ret;
}

void* null_handler(void* self, void* arg)
{
        return NULL;
}

handler_t assoc_get_handler(pair_t* funcs, int n, const char* name)
{
        handler_t func = (handler_t)assoc_get(funcs, n, name);
        return func? func: null_handler;
}

void call_by_pattern(pair_t* funcs, int n, const char* pat, void* self, void* arg)
{
        for(int i = 0; i < n; i++)
                if(match(pat, funcs[i].key))
                        ((handler_t)(funcs[i].value))(self, arg);
}

void call_by_patterns(pair_t* funcs, int n, const char* _pats, void* self, void* arg)
{
        char* pat;
        char* old_pats = strdup(_pats);
        char* pats = old_pats;
        while((pat = mystrsep(&pats, ","))){
                call_by_pattern(funcs, n, pat, self, arg);
        }
        free(old_pats);
}

enum AM_STATE {
        AM_INIT = 0,
        AM_STOP = -1,
        AM_INVALID = -2,
};

typedef struct _am_transition_t {
        long state;
        void* arg;
}am_transition_t;

typedef am_transition_t (*am_handler_t)(void* self, void* arg);

am_transition_t _am_go(am_handler_t* handlers, int state, void* self, void* arg)
{
        return handlers[state](self, arg);
}

int am_go(am_handler_t* handlers, int state, void* self, void* arg)
{
        am_transition_t transition;
        while(state != AM_STOP && state != AM_INVALID && arg != NULL){
                transition = _am_go(handlers, state, self, arg);
                state = transition.state;
                arg = transition.arg;
        }
        return state;
}

int am_go_forever(am_handler_t* handler, void* self, iter_t event_getter, void* event_src)
{
        int state = AM_INIT;
        void* arg = NULL;
        state = am_go(handler, state, self, arg);
        while(state != AM_STOP && state != AM_INVALID && (arg = event_getter(event_src))){
                state = am_go(handler, state, self, arg);
        }
        return state;
}

#ifdef TEST
// this automata will accept string such as '0*'
enum TestAmState{
        TEST_AM_INIT = 0,
        TEST_AM_GET0,
};

am_transition_t test_am_init(void* self, void* arg)
{
        return (am_transition_t){TEST_AM_GET0, NULL};
}

am_transition_t test_am_get0(void* self, char* str)
{
        fprintf(stderr, "%s\n", str);
        if(*str == '0')return (am_transition_t){TEST_AM_GET0, NULL};
        else return (am_transition_t){AM_STOP, NULL};
}

char* test_am_get_rest_str(char** str)
{
        return (*str)++;
}

long test_am()
{
        char* str = "00001";
        am_handler_t handlers[] = {(am_handler_t)test_am_init, (am_handler_t)test_am_get0};
        return am_go_forever(handlers, NULL, (iter_t)test_am_get_rest_str, &str);
}
#endif

#endif /* _CALL_H_ */
