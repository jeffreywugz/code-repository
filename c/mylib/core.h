#ifndef _CORE_H_
#define _CORE_H_

#define array_len(a) (sizeof(a)/sizeof(a[0]))
#define swap(x, y) {typeof(x) t; t=x; x=y; y=t;}

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <execinfo.h>

enum DebugLevel {PANIC, ERROR, WARNING, INFO};

#define CHECK_DEFAULT_LEVEL WARNING
#define check(level, expr, err, ...)                                    \
        _check(level, expr, err, __FILE__, __LINE__,  ##__VA_ARGS__, NULL)
#define ckerr(expr, ...) check(CHECK_DEFAULT_LEVEL, #expr, expr, ##__VA_ARGS__)
#define cktrue(expr, ...) check(CHECK_DEFAULT_LEVEL, #expr, !(expr), ##__VA_ARGS__)
void _check(int level, const char* expr, int err,
            const char* file, int line, ...);
#define show_info 0
#if show_info
#define info(format,...) fprintf(stderr, format, ##__VA_ARGS__)
#else
#define info(format,...)
#endif

#define _malloc(n) ({void* p = malloc(n); cktrue(p); p;})

void show_stackframe(int start, int num)
{
        void *trace[16];
        char **messages = (char **)NULL;
        int i, trace_size = 0;
        trace_size = backtrace(trace, 16);
        messages = backtrace_symbols(trace, trace_size);
        printf("[bt] Execution path:\n");
        for (i=start; i<trace_size && i<(start+num); ++i){
                printf("[bt] %s\n", messages[i]);
                /* free(messages[i]); */
        }
        free(messages);
}

void _check(int level, const char* expr, int err,
            const char* file, int line, ...)
{
#ifndef  NDEBUG
        const char *msg;
        va_list ap;

        if(err == 0)
                return;
        if(level < INFO)
                fprintf(stderr, "%s:%d: ", file, line);
        va_start(ap, line);
        msg = (const char*)va_arg(ap, char *);
        if (msg == NULL)
                msg = expr;
        vfprintf(stderr, msg, ap);
        fprintf(stderr, "\n");
        va_end(ap);
        if(level < WARNING){
                show_stackframe(2, 5);
                exit(-1);
        }
#endif
}

#endif /* _CORE_H_ */
