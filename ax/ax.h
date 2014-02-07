#ifndef __OB_AX_AX_H__
#define __OB_AX_AX_H__
#include "errno.h"

class AxServer
{
public:
  int init();
  int start()
  {
    int err = 0;
    return err;
  }
};

class AxClient
{
public:
  int init();
};

#endif /* __OB_AX_AX_H__ */
