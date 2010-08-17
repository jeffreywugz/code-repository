#include <stdio.h>
#include <stdarg.h>
#include <sys/time.h>
#include "util.h"

char* str_template(char* str, const char* template, ...)
{
        va_list ap;
        va_start(ap, template);
        vsprintf(str, template, ap);
        va_end(ap);
        return str;
}

long long get_time()
{
        struct timeval tv;
        if(gettimeofday(&tv, NULL)<0)
                sys_panic("gettimeofday");
        return tv.tv_sec*1000000 + tv.tv_usec;
}
