#ifndef __OB_PAXOS_PIS_H__
#define __OB_PAXOS_PIS_H__
#include <stdint.h>

struct pis_t;
typedef struct pis_t pis_t;

typedef struct pis_callback_t
{
  const char* (*cfg)(void* self, const char* key);
  void* cfg_inst;
  int (*replay)(void* self, int64_t seq, int32_t cmd, const char* clog, int64_t len);
  void* replay_inst;
} pis_callback_t;

pis_t* pis_new(pis_callback_t* callback);
int pis_start(pis_t* pis);
void pis_destroy(pis_t* pis);
int pis_propose(pis_t* pis, int32_t cmd, const char* buf, int64_t len);

#endif /* __OB_PAXOS_PIS_H__ */
