#ifdef PCOUNTER_DEF
PCOUNTER_DEF(CONN, 0)
PCOUNTER_DEF(UPKT, 0)
PCOUNTER_DEF(RPKT, 0)
PCOUNTER_DEF(IO, -1)
PCOUNTER_DEF(EPWAIT, -1)
PCOUNTER_DEF(EVWAKE, -1)
PCOUNTER_DEF(END, -1)
#endif

#ifndef __OB_AX_PCOUNTER_H__
#define __OB_AX_PCOUNTER_H__
#include "common.h"

enum
{
#define PCOUNTER_DEF(name, level) PCOUNTER_ ## name,
#include __FILE__
#undef PCOUNTER_DEF
  PCOUNTER_COUNT = PCOUNTER_END,
};

struct PCounterDesc
{
  const char* name_;
  int level_;
};
inline const PCounterDesc* get_pcounter_desc(int mod)
{
  const static PCounterDesc desc[] = {
#define PCOUNTER_DEF(name, level) {#name, level},
#include __FILE__
#undef PCOUNTER_DEF
  };
  return desc + mod;
}

class PCounterSet
{
public:
  PCounterSet(){ memset(counters_, 0, sizeof(counters_)); }
  ~PCounterSet() {}
public:
  void set(int mod, int64_t val) { AS(counters_ + mod, val); }
  void add(int mod, int64_t delta) { FAA(counters_ + mod, delta); }
  int64_t get(int mod){ return AL(counters_ + mod); }
private:
  int64_t counters_[PCOUNTER_COUNT];
};

class PCounterMonitor
{
public:
public:
  PCounterMonitor(PCounterSet& set, int64_t interval):
    pcounter_set_(set),
    report_interval_(interval),
    last_report_time_(get_us()),
    report_seq_(1)
  {
    memset(counters_, 0, sizeof(counters_));
  }
  ~PCounterMonitor() {}
public:
  void report() {
    int64_t cur_time = get_us();
    int64_t last_report_time = last_report_time_;
    if (cur_time > last_report_time + report_interval_
        && CAS(&last_report_time_, last_report_time, cur_time)
        && lock_.try_lock())
    {
      Printer printer;
      int64_t cur_set_idx = report_seq_ & 1;
      int64_t last_set_idx = cur_set_idx ^ 1;
      for(int i = 0; i < PCOUNTER_COUNT; i++)
      {
        counters_[report_seq_&1][i] = pcounter_set_.get(i);
      }
      for(int i = 0; i < PCOUNTER_COUNT; i++)
      {
        const PCounterDesc* desc = get_pcounter_desc(i);
        counters_[cur_set_idx][i] = pcounter_set_.get(i);
        if (desc->level_ >= 0)
        {
          printer.append("%s:%6ld ", desc->name_,
                         desc->level_ == 1 ? (counters_[cur_set_idx][i] - counters_[last_set_idx][i]) * 1000000/(cur_time - last_report_time): counters_[cur_set_idx][i]);
        }
      }
      MLOG(INFO, "PC[%3ld] %s", report_seq_, printer.get_str());
      report_seq_++;
      lock_.unlock();
    }
  }
private:
  PCounterSet& pcounter_set_;
  SpinLock lock_;
  int64_t report_interval_;
  int64_t last_report_time_;
  int64_t report_seq_;
  int64_t counters_[2][PCOUNTER_COUNT];
};

inline PCounterSet& get_pcounter_set() { static PCounterSet pcounter_set; return pcounter_set; }
inline PCounterMonitor& get_pcounter_monitor() { static PCounterMonitor pcounter_monitor(get_pcounter_set(), 1000000); return pcounter_monitor; }

class PCounterProfile
{
public:
  PCounterProfile(int mod): mod_(mod), start_ts_(get_us()) {}
  ~PCounterProfile() { get_pcounter_set().add(mod_, get_us() - start_ts_); }
private:
  int mod_;
  int64_t start_ts_;
};

#define PC_ADD(mod, x) get_pcounter_set().add(PCOUNTER_ ## mod, x)
#define PC_PROFILE(mod) PCounterProfile profile##mod(PCOUNTER_ ## mod)
#define PC_REPORT() get_pcounter_monitor().report()

#endif /* __OB_AX_PCOUNTER_H__ */
