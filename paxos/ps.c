#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "pis.h"

const char* _usages = "# Usages\n"
  "# set environment variable: self=ip:port group=ip1:port1,ip2:port2... workdir=.\n"
  "%1$s dumpsrc | (mkdir -p bin-src && tar jx -C bin-src)\n"
  "%1$s start\n"
  "%1$s propose ...\n";

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

void help(const char* appname)
{
  fprintf(stderr, "# compile info\n"
          "compiled by GCC %s at %s %s\n"
          "%s update at %s\n",
          __VERSION__, __DATE__, __TIME__,  __BASE_FILE__, __TIMESTAMP__);
  fprintf(stderr, _usages, appname);
}

int start_ps()
{
  int err = 0;
  pis_callback_t callback = {cfg: getcfg, cfg_inst: NULL, replay: replay, replay_inst: NULL};
  pis_t* pis = pis_new(&callback);
  if (NULL == pis)
  {
    err = -ENOMEM;
  }
  else if (0 != (err = pis_start(pis)))
  {
    error("pis_start()=>%d", err);
  }
  pis_destroy(pis);
  return 0;
}

int propose_ps()
{
  return 0;
}

extern uint8_t __src_start[]     asm("_binary_ps_tar_bz2_start");
extern uint8_t __src_end[] asm("_binary_ps_tar_bz2_end");
int dumpsrc()
{
  write(1, __src_start, __src_end - __src_start);
  return 0;
}

int main(int argc, char** argv)
{
  int err = 0;
  if (argc < 2)
  {
    err = -EINVAL;
  }
  else if (0 == strcmp(argv[1], "dumpsrc"))
  {
    err = dumpsrc();
  }
  else if (0 == strcmp(argv[1], "start"))
  {
    err = start_ps();
  }
  else if (0 == strcmp(argv[1], "propose"))
  {
    if (argc != 2)
    {
      err = -EINVAL;
    }
    else
    {
      err = propose_ps();
    }
  }
  if (-EINVAL == err)
  {
    help(argv[0]);
  }
  return err;
}
