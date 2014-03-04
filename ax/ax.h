#ifndef __OB_AX_AX_H__
#define __OB_AX_AX_H__
#include "common.h"

class AxLogServer
{
public:
  AxLogServer(): workdir_(NULL) {}
  ~AxLogServer() {
    if (NULL != workdir_)
    {
      free(workdir_);
      workdir_ = NULL;
    }
  }
public:
  int init(const char* workdir) {
    int err = AX_SUCCESS;
    if (NULL == workdir)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (NULL != workdir_)
    {
      err = AX_INIT_TWICE;
    }
    else if (NULL == (workdir_ = strdup(workdir)))
    {
      err = AX_NO_MEM;
    }
    else
    {
      MLOG(INFO, "log_server init succ, workdir=%s", workdir_);
    }
    return err;
  }
  int bootstrap() {
    int err = AX_SUCCESS;
    return err;
  }
    
  int start() {
    int err = AX_SUCCESS;
    return err;
  }
private:
  char* workdir_;
};

class AxAdminClient
{
public:
  int init() {
    int err = AX_SUCCESS;
    return err;
  }
  int start_group(Server& leader, ServerList& group) {
    int err = AX_SUCCESS;
    return err;
  }
  int stop_group(Server& leader, ServerList& group) {
    int err = AX_SUCCESS;
    return err;
  }
  int propose(Server& leader, Cursor& cursor, Buffer& content) {
    int err = AX_SUCCESS;
    return err;
  }
  int read(Server& server, Cursor& cursor, Buffer& content) {
    int err = AX_SUCCESS;
    return err;
  }
};

#endif /* __OB_AX_AX_H__ */
