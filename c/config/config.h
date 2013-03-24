
struct CfgValueGetter
{
  CfgValueGetter() {}
  virtual ~CfgValueGetter() {}
  virtual int64_t get(const char* key, int64_t default_value) = 0;
};

struct TbsysCfgValueGetter: public CfgValueGetter
{
  static const char* section;
  TbsysCfgValueGetter() {}
  ~TbsysCfgValueGetter() {}
  int64_t get(const char* key, int64_t default_value) {
    return (int64_t)TBSYS_CONFIG.getInt(section, key, default_value);
  }
};
char* TbsysCfgValueGetter::section = "";

struct CfgItem
{
  CfgItem* last_;
  CfgValueGetter* value_getter_;
  const char* name_;
  const char* key_;
  int64_t default_value_;
  int64_t value_;
  CfgItem(Cfg*& last, CfgValueGetter* value_getter, const char* name, const char* key, int64_t default_value):
    last_(last), value_getter_(value_getter), name_(name), key_(key), default_value_(default_value), value_(default_value)
  {
    last = this;
  }

  void refresh()
  {
    value_ = CfgValueGetter(key_, default_value_);
  }
  void refresh_all()
  {
    for(p = this; p != NULL; p = p->last_)
    {
      p->refresh();
    }
  }
};

#define CfgItem(name, key, value) \
  struct Cfg#name: public CfgItem { Cfg#name(): CfgItem(last_, &value_getter_, #name, #key, value){}; } name;
class GConfig
{
  public:
    static CfgItem* last_;
    static TbsysCfgValueGetter value_getter_;
    CfgItem(name, key, value);    
};
