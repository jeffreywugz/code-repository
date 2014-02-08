#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include "ax_common.h"
#include "ax_utils.h"

int Server::parse(const char* spec)
{
  int err = AX_SUCCESS;
  char* ip = NULL;
  if (2 != sscanf(spec, "%as:%u", &ip, &port_))
  {
    err = AX_INVALID_ARGUMENT;
  }
  else
  {
    ip_ = inet_addr(ip);
  }
  free(ip);
  return err;
}

int ServerList::parse(const char* spec)
{
  int err = AX_SUCCESS;
  char* str = cstrdup(__alloca__, spec);
  StrTok tokenizer(str, ",");
  for(char* tok = NULL; AX_SUCCESS == err && NULL != (tok = tokenizer.next()); )
  {
    if (count_ >= MAX_SERVER_COUNT)
    {
      err = AX_SIZE_OVERFLOW;
    }
    else
    {
      err = servers_[count_++].parse(tok);
    }
  }
  return err;
}

int Cursor::parse(const char* spec)
{
  int err = AX_SUCCESS;
  if (2 != scanf(spec, "%lu:%lu", &term_, &pos_))
  {
    err = AX_INVALID_ARGUMENT;
  }
  return err;
}

int Buffer::parse(const char* spec)
{
  int err = AX_SUCCESS;
  limit_ = used_ = strlen(spec);
  buf_ = (char*)spec;
  return err;
}

void Buffer::dump()
{
  DLOG(INFO, "dump buffer: limit=%ld used=%ld", limit_, used_);
}
