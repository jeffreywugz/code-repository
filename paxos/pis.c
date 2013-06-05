#include <errno.h>
#include <stdlib.h>
#include <pthread.h>
#include "pis.h"

typedef struct server_t
{
  int32_t ip;
  int32_t port;
} server_t;

struct pisov_t
{
  int64_t update_ts;
  char name[8];
  char token[8];
  int64_t last_piseq;
  server_t self;
  server_t master_addr;
  int64_t master_lts;
  int64_t max_clog_piseq;
  int64_t max_snapshot_piseq;
};

struct pis_t
{
  pis_callback_t* callback;
};

pis_t* pis_new(pis_callback_t* callback)
{
  pis_t* pis = NULL;
  if (NULL == callback)
  {
    //err = -EINVAL;
  }
  else if (NULL == (pis = malloc(sizeof(pis_t))))
  {
    //err = -ENOMEM;
  }
  else
  {
    pis->callback = callback;
  }
  return pis;
}

void pis_destroy(pis_t* pis)
{
  if (NULL != pis)
  {
    free(pis);
  }
}

int pis_start(pis_t* pis)
{
  return 0;
}
