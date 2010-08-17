#include <string.h>
#include <sys/time.h>
#define USER_TIME_ONLY 0
#if USER_TIME_ONLY
#include <sys/resource.h>
static void get_time(struct timeval* tv)
{
        struct rusage tp;
	getrusage(RUSAGE_SELF,&tp);
        *tv = tp.ru_utime;
}
#else
static void get_time(struct timeval* tv)
{
        gettimeofday(tv, NULL);
}
#endif

static long time_diff(struct timeval* tv1, struct timeval* tv2)
{
	return ((tv2->tv_sec - tv1->tv_sec) * 1000000 + 
		(tv2->tv_usec - tv1->tv_usec));
}

#define time_it(exp, n) ({ struct timeval tv1, tv2;     \
                        int i;                          \
                        get_time(&tv1);                 \
                        for(i = 0; i < n; i++)          \
                                exp;                    \
                        get_time(&tv2);                 \
                        (time_diff(&tv1, &tv2)/1e3/n);  \
                })


static int q2i(const char* q)
{
        unsigned int n;
        unsigned unit = 1;
        n = strlen(q);
        switch(q[n-1]){
        case 'k': unit *= (1<<10); break;
        case 'm': unit *= (1<<20); break;
        case 'g': unit *= (1<<30); break;
        default: assert(0);
        };
        return atoi(q) * unit;
}

#define MAX_N_FORMAT_CHAR 1024
static const char* i2q(int i)
{
        static char format[MAX_N_FORMAT_CHAR];
        char unit = 'B';
        assert(i >= (1<<10) && i <= (1<<30));
        if(i < (1<<10)){
                unit = 'B';
        } else if(i < (1<<20)){
                i >>= 10;
                unit = 'k';
        } else if(i < (1<<30)){
                i >>= 20;
                unit = 'm';
        } else {
                i >>= 30;
                unit = 'g';
        }
        sprintf(format, "%d%c", i, unit);
        return format;
}
