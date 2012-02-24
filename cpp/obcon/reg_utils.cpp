#include "reg_utils.h"
#include "tbsys.h"

const int MAX_N_REG_GROUP = 1<<5;
int extract_reg_match(const char* str, ObDataBuffer& buf, int max_n_match, int& n_match, regmatch_t* regmatch, char** match)
{
  int err = OB_SUCCESS;
  regmatch_t* pm = NULL;
  int64_t len = 0;
  int i = 0;
  if (NULL == str || 0 >= max_n_match || NULL == regmatch || NULL == match)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "extracrt_reg_match(str=%p, max_n_match=%d, regmatch=%p, match=%p)=>%d", str, max_n_match, regmatch, match, err);
  }
  for(i = 0; OB_SUCCESS == err && i < max_n_match; i++)
  {
    pm = regmatch + i;
    len = pm->rm_eo - pm->rm_so;
    if (0 > pm->rm_so)
    {
      break;
    }
    else if (len < 0)
    {
      err = OB_ERR_UNEXPECTED;
      TBSYS_LOG(ERROR, "pm->rm_eo[%ld] - pm->rm_so[%ld] <= 0", pm->rm_eo, pm->rm_so);
    }
    else if (buf.get_remain() <= len)
    {
      err = OB_BUF_NOT_ENOUGH;
      TBSYS_LOG(ERROR, "buf.get_remain()[%ld] <= len[%ld]", buf.get_remain(), len);
    }
    else
    {
      strncpy(buf.get_data() + buf.get_position(), str + pm->rm_so, len);
      match[i] = buf.get_data() + buf.get_position();
      buf.get_position() += len;
      *(buf.get_data() + buf.get_position()++) = 0;
    }
  }
  if (OB_SUCCESS == err)
  {
    n_match = i;
  }
  return err;
}

int reg_match(const char* pat, const char* str, ObDataBuffer& buf, int max_n_match, int& n_match, char** match)
{
  int err = OB_SUCCESS;
  int reg_err = 0;
  char reg_err_msg[256] = "no error!";
  regmatch_t regmatch[32];
  bool compiled = false;
  regex_t regex;

  for(int i = 0; i < (int)ARRAYSIZEOF(regmatch); i++)
  {
    regmatch[i].rm_so = regmatch[i].rm_eo = -1;
  }
  if (NULL == pat || NULL == str || 0 >= buf.get_remain() || 0 >= max_n_match || max_n_match > (int)ARRAYSIZEOF(regmatch))
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "reg_match(pat[%p], str[%p], buf.remain=%ld, max_n_match=%d)=>%d", pat, str, buf.get_remain(), max_n_match, err);
  }
  else if (OB_SUCCESS != (reg_err = regcomp(&regex, pat, REG_EXTENDED)))
  {
    err = OB_ERROR;
    regerror(reg_err, &regex, reg_err_msg, sizeof(reg_err_msg));
    TBSYS_LOG(ERROR, "regcomp(pat=%s, str=%s)=>%s", pat, str, reg_err_msg);
  }
  else
  {
    compiled = true;
  }
  if (OB_SUCCESS != err)
  {}
  else if (OB_SUCCESS != (reg_err = regexec(&regex, str, max_n_match, regmatch, 0)))
  {
    if(REG_NOMATCH != reg_err)
    {
      err = OB_ERROR;
      regerror(reg_err, &regex, reg_err_msg, sizeof(reg_err_msg));
      TBSYS_LOG(ERROR, "regexec(pat='%s', str='%s')=>%s", pat, str, reg_err_msg);
    }
    else
    {
      err = OB_PARTIAL_FAILED;
      TBSYS_LOG(DEBUG, "regexec(pat='%s', str='%s')=>REG_NOMATCH", pat, str);
    }
  }
  if (compiled)
  {
    regfree(&regex);
  }

  if (OB_SUCCESS == err && OB_SUCCESS != (err = extract_reg_match(str, buf, max_n_match, n_match, regmatch, match)))
  {
    TBSYS_LOG(ERROR, "extract_reg_match()=>%d", err);
  }
  return err;
}

int __reg_parse(const char* pat, const char* str, ObDataBuffer& buf, ...)
{
  int err = OB_SUCCESS;
  va_list ap;
  int n_match = 0;
  char* match[MAX_N_REG_GROUP];
  char** capture = NULL;
  if (NULL == pat || NULL == str)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "reg_test(pat=%p, str=%p)=>%d", pat, str, err);
  }
  else if (OB_SUCCESS != (err = reg_match(pat, str, buf, ARRAYSIZEOF(match), n_match, match)))
  {
    TBSYS_LOG(DEBUG, "reg_match(pat='%s', str='%s')=>%d", pat, str, err);
  }
  else
  {
    va_start(ap, buf);
    for (int i = 1; i < n_match && (capture = va_arg(ap, char**)); i++)
    {
      *capture = match[i];
    }
    va_end(ap);
  }
  return err;
}

int reg_helper(const char* pat, const char *str, int& n_match, char**& match)
{
  int err = OB_SUCCESS;
  static __thread char cbuf[2048];
  ObDataBuffer buf(cbuf, sizeof(cbuf));
  static __thread char* tmp_match[32];
  if (NULL == pat || NULL == str)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "reg_test(pat=%p, str=%p)=>%d", pat, str, err);
  }
  else if (OB_SUCCESS != (err = reg_match(pat, str, buf, ARRAYSIZEOF(tmp_match), n_match, tmp_match)))
  {
    TBSYS_LOG(DEBUG, "reg_match(pat='%s', str='%s')=>%d", pat, str, err);
  }
  else
  {
    match = tmp_match;
  }
  return err;
}

int reg_test(const char* pat, const char* str)
{
  int err = OB_SUCCESS;
  int n_match = 0;
  char** match = NULL;
  if (OB_SUCCESS != (err = reg_helper(pat, str, n_match, match)))
  {
    TBSYS_LOG(ERROR, "reg_helper(pat='%s', str='%s')=>%d", pat, str, err);
  }
  else
  {
    TBSYS_LOG(INFO, "n_match=%d", n_match);
    for (int i = 0; i < n_match; i++)
    {
      TBSYS_LOG(INFO, "match[%d]='%s'", i, match[i]);
    }
  }
  return err;
}
