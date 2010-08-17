#ifndef  __UTIL_H__
#define  __UTIL_H__

#define MEM_ERR 1
#define ARG_ERR 2
#define EOF_ERR 3

#define  max(a, b) ((a)>=(b)?(a):(b))
#define  swap(a, b) { typeof(a) c; c=a; a=b; b=c;}

void fatal(char* msg, int code);

#endif  /*__UTIL_H__*/
