#include <string.h>
#include "utils.h"

Alloca __alloca__;

char* safe_strncpy(char* dest, const char* src, int64_t len)
{
  strncpy(dest, src, len);
  dest[len - 1] = 0;
  return dest;
}
