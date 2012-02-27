#include <string.h>
#include <ctype.h>
#include <fnmatch.h>
#include <regex.h>
#include "common.h"
#include "str.h"

char* sf(char* buf, const int64_t len, int64_t& pos, const char* format, va_list ap)
{
  char* result = NULL;
  int64_t str_len = 0;
  if (!check_buf(buf, len, pos) || NULL == format)
  {}
  else if (0 > (str_len = vsnprintf(buf + pos, len - pos, format, ap)))
  {}
  else
  {
    result = buf + pos;
    pos += str_len + 1;
  }
  return result;
}

char* sf(char* buf, const int64_t len, int64_t& pos, const char* format, ...)
{
  va_list ap;
  char* result = NULL;
  va_start(ap, format);
  result = sf(buf, len, pos, format, ap);
  va_end(ap);
  return result;
}

static void hex_encode_char(char* out, const char c)
{
        static const char* map = "0123456789ABCDEF";
        *out++ = map[c & 0x0f];
        *out++ = map[(c & 0xf0)>>4];
}

char* str_escape(char* out, const char* in, const int64_t len)
{
  int err = 0;
  char* result = out;
  int64_t i = len;
  if (NULL == out || NULL == in || 0 > len)
  {
    err = EINVAL;
  }
  while(0 == err && i--)
  {
    if(isgraph(*in) && *in != '\\')
    {
      *out++ = *in++;
    }
    else
    {
      *out++ = '\\';
      if(*in == '\\')*out++ = '\\';
      else if(*in == ' ')*out++ = 's';
      else if(*in == '\f')*out++ = 'f';
      else if(*in == '\t')*out++ = 't';
      else if(*in == '\v')*out++ = 'v';
      else if(*in == '\n')*out++ = 'n';
      else if(*in == '\r')*out++ = 'r';
      else *out++ = 'x', hex_encode_char(out, *in), out += 2;
      in++;
    }
  }
  if (0 == err)
  {
    *out = 0;
  }
  else
  {
    result = NULL;
  }
  return result;
}

char* strncpy2(char* buf, const int64_t len, int64_t& pos, const char* src, const int64_t max_size)
{
  char* result = NULL;
  int64_t copy_len = min(len - pos - 1, max_size);
  if (!check_buf(buf, len, pos))
  {}
  else if (NULL == (result = strncpy(buf + pos, src, copy_len)))
  {
  }
  else
  {
    pos += copy_len + 1;
  }
  if (NULL != buf && copy_len > 0)
  {
    buf[pos-1] = 0;
  }
  return result;
}

char* mystrsep(char** str, const char* delim)
{
        char* start;
        if(*str == NULL)return NULL;
        *str += strspn(*str, delim);
        start = *str;
        *str += strcspn(*str, delim);
        if(**str == 0)*str = NULL;
        else *(*str)++ = 0;
        return *start? start: NULL;
}

bool glob_match(const char* pattern, const char *str)
{
  return !fnmatch(pattern, str, 0);
}

const int MAX_N_REG_GROUP = 1<<5;
int extract_reg_match(const char* str, char* buf, const int64_t len, int64_t& pos,
                      int max_n_match, int& n_match, regmatch_t* regmatch, char** match)
{
  int err = 0;
  regmatch_t* pm = NULL;
  int64_t match_len = 0;
  int i = 0;
  if (NULL == str || !check_buf(buf, len, pos) || 0 > max_n_match || NULL == regmatch || NULL == match)
  {
    err = EINVAL;
  }
  for(i = 0; 0 == err && i < max_n_match; i++)
  {
    pm = regmatch + i;
    match_len = pm->rm_eo - pm->rm_so;
    if (0 > pm->rm_so)
    {
      break;
    }
    else if (match_len < 0)
    {
      err = -1;
    }
    else if (pos + match_len >= len)
    {
      err = ENOBUFS;
    }
    else if (NULL == (match[i] = strncpy2(buf, len, pos, str + pm->rm_so, match_len)))
    {
      err = -1;
    }
  }
  if (0 == err)
  {
    n_match = i;
  }
  return err;
}

int reg_match(const char* pat, const char* str, char* buf, const int64_t len, int64_t& pos,
              int max_n_match, int& n_match, char** match)
{
  int err = 0;
  int reg_err = 0;
  regmatch_t regmatch[32];
  bool compiled = false;
  regex_t regex;

  for(int i = 0; i < (int)array_len(regmatch); i++)
  {
    regmatch[i].rm_so = regmatch[i].rm_eo = -1;
  }
  if (NULL == pat || NULL == str || NULL == buf || 0 > len || 0 > pos || pos > len
      || 0 >= max_n_match || max_n_match > (int)array_len(regmatch))
  {
    err = EINVAL;
  }
  else if (0 != (err = reg_err = regcomp(&regex, pat, REG_EXTENDED)))
  {}
  else
  {
    compiled = true;
  }
  if (0 == err && 0 != (err = reg_err = regexec(&regex, str, max_n_match, regmatch, 0)))
  {}
  if (compiled)
  {
    regfree(&regex);
  }

  if (0 == err && 0 != (err = extract_reg_match(str, buf, len, pos, max_n_match, n_match, regmatch, match)))
  {
  }
  char reg_err_msg[256] = "no error!";
  if (0 != reg_err)
  {
    regerror(reg_err, &regex, reg_err_msg, sizeof(reg_err_msg));
    fprintf(stderr, "RegErr: pat='%s', str='%s', err='%s'\n", pat, str, reg_err_msg);
  }
  return err;
}

int __reg_parse(const char* pat, const char* str, char* buf, const int64_t len, int64_t& pos, ...)
{
  int err = 0;
  va_list ap;
  int n_match = 0;
  char* match[MAX_N_REG_GROUP];
  char** capture = NULL;
  if (NULL == pat || NULL == str || !check_buf(buf, len, pos))
  {
    err = EINVAL;
  }
  else if (0 != (err = reg_match(pat, str, buf, len, pos, array_len(match), n_match, match)))
  {
  }
  else
  {
    va_start(ap, pos);
    for (int i = 1; i < n_match && (capture = va_arg(ap, char**)); i++)
    {
      *capture = match[i];
    }
    va_end(ap);
  }
  return err;
}

#define reg_parse(pat, str, buf, len, pos, ...) __reg_parse(pat, str, buf, len, pos, ##__VA_ARGS__, NULL)
int reg_helper(const char* pat, const char *str, int& n_match, char**& match)
{
  int err = 0;
  static __thread char cbuf[2048];
  int64_t pos = 0;
  static __thread char* tmp_match[32];
  if (NULL == pat || NULL == str)
  {
    err = EINVAL;
  }
  else if (0 != (err = reg_match(pat, str, cbuf, sizeof(cbuf), pos, array_len(tmp_match), n_match, tmp_match)))
  {
  }
  else
  {
    match = tmp_match;
  }
  return err;
}

int reg_test(const char* pat, const char* str)
{
  int err = 0;
  int n_match = 0;
  char** match = NULL;
  if (0 != (err = reg_helper(pat, str, n_match, match)))
  {
    log(DEBUG, "reg_helper(pat='%s', str='%s')=>%d\n", pat, str, err);
  }
  else
  {
    printf("n_match=%d\n", n_match);
    for (int i = 0; i < n_match; i++)
    {
      printf("match[%d]='%s'\n", i, match[i]);
    }
  }
  return err;
}

#ifdef __TEST_UTILS_STR__
#include <assert.h>
int main(int argc, char *argv[])
{
  char cbuf[1<<10];
  int64_t pos = 0;
  char* svr1 = NULL;
  char* svr2 = NULL;
  assert(0 == reg_test("init +([a-zA-Z0-9._:]+) ([a-zA-Z0-9._:]+)", "init 10.232.35.40:8080 10.232.35.40:8000"));
  assert(REG_NOMATCH == reg_test("init +([a-zA-Z0-9._:]+) ([a-zA-Z0-9._:]+)", "init 10.232.35.40:8080, 10.232.35.40:8000"));
  assert(0 == reg_parse("init +([a-zA-Z0-9._:]+) ([a-zA-Z0-9._:]+)", "init 10.232.35.40:8080 10.232.35.40:8000", cbuf, sizeof(cbuf), pos, &svr1, &svr2));
  printf("svr1:%s\n", svr1);
  printf("svr2:%s\n", svr2);
  return 0;
}
#endif
