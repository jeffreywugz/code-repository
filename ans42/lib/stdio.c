#include <stdarg.h>
#include <stdio.h>

#ifdef __LIBRARY__
/* int printf(const char *fmt, ...) */
/* { */
/*         va_list args; */
/*         int i; */
/*         va_start(args, fmt); */
/*         i=vsprintf(buf,fmt,args); */
/*         va_end(args); */
/*         write(buf); */
/*         return i; */
/* } */
#else
#include <screen.h>

static char buf[1024];

int printf(const char *fmt, ...)
{
        va_list args;
        int i;
        va_start(args, fmt);
        i=vsprintf(buf,fmt,args);
        va_end(args);
        puts(buf);
        return i;
}

int sprintf(char *buf, const char *fmt, ...)
{
        va_list args;
        int i;
        va_start(args, fmt);
        i=vsprintf(buf,fmt,args);
        va_end(args);
        return i;
}
        
int vprintf(const char *fmt, va_list args)
{
        int i;
        i=vsprintf(buf,fmt,args);
        puts(buf);
        return i;
}

void message(int level, const char *fmt, ...)
{
        va_list args;
        if(level > MSG_LEVEL)
                return;
        va_start(args, fmt);
        vprintf(fmt,args);
        va_end(args);
}

#endif

