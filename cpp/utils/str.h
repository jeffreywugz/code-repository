#ifndef __UTILS_STR_UTILS_H__
#define __UTILS_STR_UTILS_H__
#include <stdint.h>
#include <stdarg.h>

char* sf(char* buf, const int64_t len, int64_t& pos, const char* format, va_list ap);
char* sf(char* buf, const int64_t len, int64_t& pos, const char* format, ...)
  __attribute__ ((format (printf, 4, 5)));
char* str_escape(char* out, const char* in, const int64_t len);
char* strncpy2(char* buf, const int64_t len, int64_t& pos, const char* src, const int64_t max_size);
char* mystrsep(char** str, const char* delim);
bool glob_match(const char* pattern, const char *str);
int reg_match(const char* pat, const char* str, char* buf, const int64_t len, int64_t& pos,
              int max_n_match, int& n_match, char** match);
int __reg_parse(const char* pat, const char* str, char* buf, const int64_t len, int64_t& pos, ...);
#define reg_parse(pat, str, buf, len, pos, ...) __reg_parse(pat, str, buf, len, pos, ##__VA_ARGS__, NULL)
#endif /* __UTILS_STR_UTILS_H__ */
