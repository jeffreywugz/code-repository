#include <errno.h>
#include <stdint.h>

int sub(int64_t len, char* buf, const char* tpl, int64_t n, const char* args[])
{
  int err = 0;
  int64_t idx = 0;
  char* p = NULL;
  if (NULL == buf || NULL == tpl || NULL == args || len <= 0)
  {
    err = -EINVAL;
  }
  for(p = buf; OB_SUCCESS == err && *tpl && p < buf + len;)
  {
    if (*tpl == '$'){
      idx = *++tpl - 'a';
      if (*tpl != 0)
        tpl++;
      if (0 > idx || idx >= n){
        fprintf(stderr, "invalid idx[%ld], n=%ld\n", idx, n);
      }
      else if (NULL == args[idx])
      {
        err = -EINVAL;
        fprintf(stderr, "args[%ld] == NULL\n", idx);
      }
      else if (0 != args[idx][n = strnlen(args[idx], buf + len - p - 1)])
      {
        err = -ENOMEM;
        fprintf(stderr, "args[%ld] too long to append, remain=%ld\n", idx, buf + len - p - 1);
      }
      else
      {
        memcpy(p, args[idx], n);
        p += n;
      }
    } else {
      *p++ = *tpl++;
    }
  }
  if (*tpl != 0)
  {
    err = -EINVAL;
  }
  else if (p >= buf + len)
  {
    err = -ENOMEM;
  }
  else
  {
    *p = 0;
  }
  return err;
}

