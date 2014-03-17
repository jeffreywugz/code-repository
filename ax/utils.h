#ifndef __OB_AX_AX_UTILS_H__
#define __OB_AX_AX_UTILS_H__
#include "common.h"

class Alloca
{
public:
  Alloca() {}
  ~Alloca() {}
  void free(void* p) { UNUSED(p); }
  void* alloc(const int64_t size){ return alloca(size); }
};
Alloca __alloca__ WEAK_SYM;

inline char* safe_strncpy(char* dest, const char* src, int64_t len)
{
  strncpy(dest, src, len);
  dest[len - 1] = 0;
  return dest;
}

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

struct TimerReporter
{
  TimerReporter(const char* name, int64_t interval): name_(name), report_interval_(interval), last_report_ts_(0), last_report_value_(0), last_report_count_(0) {}
  ~TimerReporter(){}
  void add(int64_t cur_ts, int64_t value, int64_t count) {
    int64_t last_report_ts = last_report_ts_;
    if (cur_ts > last_report_ts + report_interval_ && __sync_bool_compare_and_swap(&last_report_ts_, last_report_ts, cur_ts))
    {
      if (last_report_ts > 0)
      {
        MLOG(INFO, "%s value/count=%ld  count/time=%ld value/time=%lf%%", name_,
                  (value - last_report_value_)/(count - last_report_count_ + 1),
                  1000000 * (count - last_report_count_)/(cur_ts - last_report_ts),
                  (double)100.0 * (double)(value-last_report_value_)/(double)(cur_ts - last_report_ts));
      }
      last_report_value_ = value;
      last_report_count_ = count;
      last_report_ts = cur_ts;
    }
  }
  const char* name_;
  int64_t report_interval_;
  int64_t last_report_ts_;
  int64_t last_report_value_;
  int64_t last_report_count_;
};

struct Profiler
{
  Profiler(const char* name): reporter_(name, 1000000), total_time_(0), total_count_(0) {}
  ~Profiler() {}
  int64_t& get_tlps_time() { // thread local profile start
    static __thread int64_t time = 0;
    return time;
  }
  void profile_start() {
    get_tlps_time() = get_us();
  }
  void profile_end() {
    int64_t end_time = get_us();
    reporter_.add(end_time, __sync_add_and_fetch(&total_time_, end_time - get_tlps_time()), __sync_add_and_fetch(&total_count_, 1));
  }
  void hit() {
    reporter_.add(get_us(), 0, __sync_add_and_fetch(&total_count_, 1));
  }
  TimerReporter reporter_;
  int64_t total_time_;
  int64_t total_count_;
};

#define HIT_PROFILE(id) {static Profiler profiler(id); profiler.hit(); }
#endif /* __AX_AX_AX_UTILS_H__ */
