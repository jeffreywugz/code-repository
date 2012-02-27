#ifndef __UTILS_COMMON_H__
#define __UTILS_COMMON_H__
#include <stdint.h>
#include <errno.h>
#include <stdio.h>
#include <stdarg.h>
#include <sys/time.h>

#define EERR 1001
#define EINIT 1001
#define EINTERNAL 1002
#define ESTR 1003
#define EEND 1004
#define DECLARE_COPY_AND_ASSIGN(TypeName)       \
  TypeName(const TypeName&);                    \
  void operator=(const TypeName&);
#define array_len(A) (sizeof(A)/sizeof(A[0]))

#define MAX_STR_BUF_SIZE 1024

template<typename T>
T max(T x, T y)
{
  return x > y? x: y;
}

template<typename T>
T min(T x, T y)
{
  return x > y? y: x;
}

inline const char* str_bool(const bool b)
{
  return b? "true": "false";
}

inline bool check_buf(const char* buf, const int64_t len, const int64_t pos)
{
  return NULL != buf && 0 <= len && 0 <= pos && pos <= len;
}

enum DebugLevel {PANIC, ERROR, WARN, INFO, DEBUG};
extern int __global_log_level__;
inline const char* get_log_level_name(const int level)
{
  const char* level_name[] = {"PANIC", "ERROR", "WARN", "INFO", "DEBUG"};
  return (level < 0 || level >= (int)array_len(level_name))? "UNKNOWN": level_name[level];
}

#define log(level, format, ...)                                    \
  __log(level, __FILE__, __LINE__,  format, ##__VA_ARGS__)
inline int __log(int level, const char* file, int line, const char* format, ...)
{
        va_list ap;
        if(level > __global_log_level__)
          return 0;
        va_start(ap, format);
        fprintf(stderr, "%s:%d: %s: ", file, line, get_log_level_name(level));
        vfprintf(stderr, format, ap);
        fprintf(stderr, "\n");
        va_end(ap);
        return 0;
}
inline  int64_t get_usec()
{
  struct timeval time_val;
  gettimeofday(&time_val, NULL);
  return time_val.tv_sec*1000000 + time_val.tv_usec;
}

#define profile(expr) ({                                                \
      int64_t old_us = 0;                                               \
      int64_t new_us = 0;                                               \
      int64_t result = 0;                                               \
      old_us = get_usec();                                              \
      result = expr;                                                    \
      new_us = get_usec();                                              \
      printf("%s=>%ld in %ldms\n", #expr, result, new_us - old_us);     \
      new_us - old_us; })
#endif /* __UTILS_COMMON_H__ */
