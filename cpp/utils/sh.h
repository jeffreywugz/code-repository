#ifndef __UTILS_SH_UTILS_H__
#define __UTILS_SH_UTILS_H__

#include <stdlib.h>
#include "func.h"
#include "str.h"

int sh(const char* cmd);
char* popen(char* buf, const int64_t len, int64_t& pos, const char* cmd);
struct MgTool {
  const static int64_t DEFAULT_CBUF_LEN = 1<<12;
  CBuf cbuf;
  BufHolder buf_holder;
  MgTool(int64_t limit=DEFAULT_CBUF_LEN) {
    cbuf.set(buf_holder.get_buf(limit), limit);
  }
  ~MgTool() {
  }

  int reset() {
    int err = 0;
    cbuf.pos_ = 0;
    return err;
  }
    
  char* sf(const char* format, ...) {
    va_list ap;
    char* result = NULL;
    va_start(ap, format);
    result = ::sf(cbuf.buf_, cbuf.len_, cbuf.pos_, format, ap);
    va_end(ap);
    return result;
  }

  int sh(const char* format, ...) {
    va_list ap;
    int err = 0;
    va_start(ap, format);
    err = ::sh(::sf(cbuf.buf_, cbuf.len_, cbuf.pos_, format, ap));
    va_end(ap);
    return err;
  }

  char* popen(const char* format, ...) {
    va_list ap;
    char* result = NULL;
    va_start(ap, format);
    result = ::popen(cbuf.buf_, cbuf.len_, cbuf.pos_, ::sf(cbuf.buf_, cbuf.len_, cbuf.pos_, format, ap));
    va_end(ap);
    return result;
  }
};

int read_file(const char* path, char* buf, const int64_t buf_size, int64_t& read_bytes);
int read_line(char* buf, const int64_t buf_size, int64_t& read_bytes, const char* prompt);

class Interpreter
{
  public:
    Interpreter(){}
    virtual ~Interpreter(){}
    virtual int execute(const char* cmd) = 0;
};

int execute_str(Interpreter* interpreter, const char* str, bool keep_going, const char* delim);
int execute_file(Interpreter* interpreter, const char* path, bool keep_going, const char* delim);
int execute_interactive(Interpreter* interpreter, bool keep_going, const char* delim, const char* prompt);

class CmdArgsParser
{
  const static int64_t MAX_N_ARGS = 1<<10;
  struct arg_t {
    arg_t(): name_(NULL), value_(NULL), default_value_(NULL) {}
    ~arg_t() {}
    const char* name_;
    const char* value_;
    const char* default_value_;
  };
  public:
    CmdArgsParser(): parse_seq_(0), n_args_(0) {
      default_arg_.name_ = "*default*";
      default_arg_.value_ = null_;
      default_arg_.default_value_ = null_;
    }
    ~CmdArgsParser() {}
    bool reset() {
      memset(args_, 0, sizeof(args_));
      n_args_ = 0;
      parse_seq_ |= 1;
      return true;
    }
    bool check(int argc, char** argv, ...) {
      bool args_is_valid = true;
      char* p = NULL;
      parse_seq_ = (parse_seq_&~1) + 2;

      for(int64_t i = 0; i < n_args_/2; i++) {
        arg_t arg = args_[i];
        args_[i] = args_[n_args_-1-i];
        args_[n_args_-1-i] = arg;
      }
      for(int64_t i = 0; i < argc; i++) {
        if (argv[i][0] == ':' || NULL == (p = strchr(argv[i], '=')))continue;
        *p++ = 0;
        arg_t* arg = get_arg(argv[i]);
        if (arg && &default_arg_ != arg) arg->value_ = p;
      }
      for(int64_t i = 0; i < argc; i++) {
        if (argv[i][0] != ':' && (p = strchr(argv[i], '=')))continue;
        p = argv[i][0] == ':'? argv[i]+1: argv[i];
        arg_t* arg = get_next_unset_arg();
        if (arg && arg->name_) arg->value_ = p;
      }
      for(int64_t i = 0; i < n_args_; i++) {
        args_[i].value_ = args_[i].value_?: args_[i].default_value_;
        if (null_ == args_[i].value_)args_is_valid = false;
      }
      dump(argc, argv);
      return args_is_valid;
    }

    void dump(int argc, char** argv) {    
      printf("cmd_args_parser.dump:\n");
      for(int64_t i = 0; i < argc; i++) {
        printf("argv[%ld]=%s\n", i, argv[i]);
      }
      for(int64_t i = 0; i < n_args_; i++) {
        printf("args[%ld]={name=%s, value=%s, default=%s}\n",
               i, args_[i].name_, args_[i].value_, args_[i].default_value_);
      }
    }

    arg_t* get_next_unset_arg() {
      for(int64_t i = 0; i < n_args_; i++) {
        if (null_ == args_[i].value_)
          return args_ + i;
      }
      return NULL;
    }

    arg_t* get_arg(const char* name, const char* default_value = NULL) {
      assert(n_args_ < MAX_N_ARGS && name);
      if (parse_seq_&1) {
        args_[n_args_].name_ = name;
        args_[n_args_].default_value_ = default_value;
        args_[n_args_].value_ = null_;
        return args_ + (n_args_++);
      }
      for(int64_t i = 0; i < n_args_; i++) {
        if (0 == strcmp(args_[i].name_, name))
          return args_ + i;
      }
      return &default_arg_;
    }
  private:
    static const char* null_;
    int64_t parse_seq_;
    int64_t n_args_;
    arg_t default_arg_;
    arg_t args_[MAX_N_ARGS];    
};
const char* CmdArgsParser::null_ = "*null*";
CmdArgsParser __cmd_args_parser;
#define _Arg(name, ...) __cmd_args_parser.get_arg(#name,  ##__VA_ARGS__)
#define IntArg(name, ...) atoll(__cmd_args_parser.get_arg(#name,  ##__VA_ARGS__)->value_)
#define StrArg(name, ...) __cmd_args_parser.get_arg(#name,  ##__VA_ARGS__)->value_
#define CmdCall(argc, argv, func, ...) \
  (argc >= 2 && !strcmp(#func, argv[1]) && __cmd_args_parser.reset() && __cmd_args_parser.check(argc-2, argv+2, ##__VA_ARGS__))? \
  func(__VA_ARGS__)
#endif /* __UTILS_SH_UTILS_H__ */
