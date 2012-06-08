#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "common/ob_define.h"
#include "common/ob_server.h"
#include "common/utility.h"

using namespace oceanbase::common;
int to_server(ObServer& server, const char* spec)
{
  int err = OB_SUCCESS;
  char* p = NULL;
  char ip[64] = "";
  int32_t port = 0;
  if (NULL == spec)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "spec == NULL");
  }
  else if (NULL == (p = strchr(spec, ':')))
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "strchr(spec='%s', ':')=>NULL", spec);
  }
  else 
  {
    strncpy(ip, spec, min(p - spec, (int64_t)sizeof(ip)));
    port = atoi(p+1);
  }
  if (OB_SUCCESS != err)
  {}
  else if (0 >= port)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "to_server(spec=%s)=>%d", spec, err);
  }
  else
  {
    TBSYS_LOG(INFO, "to_server(ip=%s, port=%d)=>%d", ip, port, err);
    server.set_ipv4_addr(ip, port);
  }
  return err;
}

ObServer _gsvr1;
ObServer _gsvr2;

const char my_interp[] __attribute__((section(".interp")))
    = "/lib64/ld-linux-x86-64.so.2";

void debug_loop()
{
  while(1) {
    pause();
    // inspect process status here using gdb
  }
}

typedef void*(*pthread_handler_t)(void*);
void __attribute__((constructor)) so_init()
{
  pthread_t thread;
  printf("debug.so init\n");
  pthread_create(&thread, NULL, (pthread_handler_t)debug_loop, NULL);
}

int mystart(int argc, char** argv)
{
  printf("debug.so\n");
  exit(0);
}
