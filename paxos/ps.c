#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include "pis.h"

const char* _usages = "Usages\n"
  "init_pg_members='' workdir='' ./%1$";

#define UNUSED(v) ((void)(v))
#define __log_format(format, ...) "%s:%d " format "\n", __FILE__, __LINE__, __VA_ARGS__
#define error(format, ...) fprintf(stderr, "ERROR" __log_format(format, __VA_ARGS__))
const char* getcfg(void* arg, const char* key)
{
  UNUSED(arg);
  return getenv(key);
}

int replay(void* self, int64_t seq, int32_t cmd, const char* clog, int64_t len)
{
  fprintf(stderr, "replay(seq=%ld, cmd=%d, clog=%p[%ld]\n", seq, cmd, clog, len);
  return 0;
}

void print_compile_info()
{
  fprintf(stderr, "compiled by GCC %s at %s %s\n"
          "%s update at %s\n",
          __VERSION__, __DATE__, __TIME__,  __BASE_FILE__, __TIMESTAMP__);
  fprintf();
}
int main()
{
  int err = 0;
  pis_callback_t callback = {cfg: getcfg, cfg_inst: NULL, replay: replay, replay_inst: NULL};
  pis_t* pis = pis_new(&callback);
  print_compile_info();
  if (NULL == pis)
  {
    err = -ENOMEM;
  }
  else if (0 != (err = pis_start(pis)))
  {
    error("pis_start()=>%d", err);
  }
  pis_destroy(pis);
  return err;
}
