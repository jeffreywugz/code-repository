#ifndef __OB_AX_AX_UTILS_H__
#define __OB_AX_AX_UTILS_H__
#include "ax_common.h"

class Alloca
{
public:
  Alloca() {}
  ~Alloca() {}
  void free(void* p) { UNUSED(p); }
  void* alloc(const int64_t size){ return alloca(size); }
};
extern Alloca __alloca__;

template<typename Allocator, typename T>
T* new_obj(Allocator& allocator, T*& obj)
{
  if (NULL == (obj = (T*)allocator.alloc(sizeof(T))))
  {}
  else
  {
    new(obj)T();
  }
  return obj;
}

template<typename Allocator>
char* cstrdup(Allocator& allocator, const char* str_)
{
  char* str = NULL;
  if (NULL == str_)
  {}
  else if (NULL == (str = (char*)allocator.alloc(strlen(str_) + 1)))
  {}
  else
  {
    strcpy(str, str_);
  }
  return str;
}

template <typename Allocator>
char* bufdup(Allocator& allocator, const char* src, int64_t len)
{
  int err = AX_SUCCESS;
  char* dest = NULL;
  if (NULL == src || len <= 0)
  {
    err = AX_INVALID_ARGUMENT;
  }
  else if (NULL == (dest = reinterpret_cast<char*>(allocator.alloc(len))))
  {
    err = AX_NO_MEM;
  }
  else
  {
    memcpy(dest, src, len);
  }
  return AX_SUCCESS == err? dest: NULL;
}

class StrTok
{
  public:
    StrTok(char* str, const char* delim): str_(str), delim_(delim), savedptr_(NULL) {}
    ~StrTok(){}
    char* next() {
      if (NULL == savedptr_)
      {
        return strtok_r(str_, delim_, &savedptr_);
      }
      else
      {
        return strtok_r(NULL, delim_, &savedptr_);
      }
    }
  private:
    char* str_;
    const char* delim_;
    char* savedptr_;
};
#endif /* __AX_AX_AX_UTILS_H__ */
