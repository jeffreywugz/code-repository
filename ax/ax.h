#ifndef __OB_AX_LOG_SERVER_H__
#define __OB_AX_LOG_SERVER_H__
#include "common.h"

#define MAX_SERVER_COUNT 16
struct ServerList
{
  ServerList(): count_(0) {}
  ~ServerList() {}
  int parse(const char* spec) {
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
  int count_;
  Server servers_[MAX_SERVER_COUNT];
};

typedef uint64_t Term;
typedef uint64_t Pos;
struct Cursor
{
  Cursor(): term_(), pos_() {}
  ~Cursor() {}
  int parse(const char* spec) {
    int err = AX_SUCCESS;
    if (2 != scanf(spec, "%lu:%lu", &term_, &pos_))
    {
      err = AX_INVALID_ARGUMENT;
    }
    return err;
  }
  Cursor& set(Term term, Pos pos) {
    term_ = term;
    pos_ = pos;
    return *this;
  }
  bool equal(const Cursor& that) const { return this->term_ == that.term_ && this->pos_ == that.pos_; }
  int compare(const Cursor& that) const {
    int ret = 0;
    if (this->term_ > that.term_)
    {
      ret = 1;
    }
    else if (this->term_ < that.term_)
    {
      ret = -1;
    }
    else if (this->pos_ > that.pos_)
    {
      ret = 1;
    }
    else if (this->pos_ < that.pos_)
    {
      ret = -1;
    }
    else
    {
      ret = 0;
    }
    return ret;
  }
  Term term_;
  Pos pos_;
};

struct Token
{
  Token(): server_(), start_time_(0) {}
  ~Token() {}
  const char* repr(Printer& printer) const { return printer.new_str("{server=%s,start_time=%ld}", server_.repr(printer), start_time_); }
  void reset() { server_.reset(); start_time_ = 0; }
  bool is_valid() const { return server_.is_valid() && start_time_ > 0; }
  Server server_;
  uint64_t start_time_;
};

#define MAX_TOKEN_COUNT 16
struct GroupConfig
{
  GroupConfig(): group0_mask_(0), group1_mask_(0) {}
  ~GroupConfig() {}
  GroupConfig& make_first_group(Token token, Cursor cursor)
  {
    group0_mask_ = 0x1;
    group1_mask_ = 0;
    group0_born_cursor_ = cursor;
    token_array_[0] = token;
    return *this;
  }
  Cursor group0_born_cursor_;
  Cursor group1_born_cursor_;
  uint32_t group0_mask_;
  uint32_t group1_mask_;
  Token token_array_[MAX_TOKEN_COUNT];
};

struct VoteReq
{
  Term cur_term_;
  Cursor max_cursor_;
  Token candinate_;
};

struct AppendReq
{
  AppendReq() {}
  ~AppendReq() {}
  const char* repr(Printer& printer) const {
    return printer.new_str("{term=%ld}", cur_term_);
  }
  AppendReq& make_first_req(Token token) {
    start_cursor_.set(0, 0);
    end_cursor_.set(1, 1);
    commit_cursor_.set(1, 1);
    group_config_.make_first_group(token, commit_cursor_);
    return *this;
  }
  Term cur_term_;
  Cursor start_cursor_;
  Cursor end_cursor_;
  Cursor commit_cursor_;
  GroupConfig group_config_;
  Buffer log_data_;
};

struct Ballot
{
  Term cur_term_;
  Token self_;
  Token leader_;
  Cursor local_end_cursor_;
  Cursor commit_cursor_;
  GroupConfig group_config_;
  Ballot(){}
  ~Ballot(){}
  int format(Token* token) {
    int err = AX_SUCCESS;
    cur_term_ = 1;
    self_ = *token;
    local_end_cursor_.set(1, 0);
    commit_cursor_.set(1, 0);
    return err;
  }
  int check_term(Term term) {
    int err = AX_SUCCESS;
    if (cur_term_ > term)
    {
      err = AX_TERM_STALE;
    }
    else if (cur_term_ < term)
    {
      cur_term_ = term;
      leader_.reset();
    }
    return err;
  }
  int update(Cursor end_cursor, GroupConfig& grpcfg) {
    int err = AX_SUCCESS;
    local_end_cursor_ = end_cursor;
    group_config_ = grpcfg;
    return err;
  }
  int vote(Term term, Cursor end_cursor, Token cadinate) {
    int err = AX_SUCCESS;
    if (term != cur_term_)
    {
      err = AX_TERM_NOT_MATCH;
    }
    else if (local_end_cursor_.compare(end_cursor) > 0)
    {
      err = AX_CURSOR_LAGGED;
    }
    else if (leader_.is_valid())
    {
      err = AX_VOTE_OTHER;
    }
    else
    {
      leader_ = cadinate;
    }
    return err;
  }
};

class BallotStore
{
public:
  BallotStore() {}
  ~BallotStore() {}
public:
  int format(const char* path, Token* token) {
    int err = AX_SUCCESS;
    MLOG(INFO, "ballot format: path='%s' token=%s", path, repr(*token));
    err = ballot_.format(token);
    return err;
  }
  int open(const char* path) {
    int err = AX_SUCCESS;
    MLOG(INFO, "ballot open: path=%s", path);
    return err;
  }
  int close() {
    int err = AX_SUCCESS;
    MLOG(INFO, "ballot close");
    return err;
  }
  Token get_self() { return ballot_.self_; }
  int load(Ballot* ballot) {
    int err = AX_SUCCESS;
    *ballot = ballot_;
    return err;
  }
  int store(Ballot* ballot) {
    int err = AX_SUCCESS;
    ballot_ = *ballot;
    return err;
  }
private:
  Ballot ballot_;
};

class LogStore
{
public:
  enum {LOG_BUF_LIMIT = 1<<24};
public:
  LogStore() {}
  ~LogStore() {}
public:
  int format(const char* path) {
    int err = AX_SUCCESS;
    MLOG(INFO, "log_store format: path='%s'", path);
    return err;
  }
  int open(const char* path) {
    int err = AX_SUCCESS;
    MLOG(INFO, "log_store open: path='%s'", path);
    return err;
  }
  int close() {
    int err = AX_SUCCESS;
    MLOG(INFO, "log_store close");
    return err;
  }
  int append(AppendReq& req) {
    int err = AX_SUCCESS;
    MLOG(INFO, "log_store append:%s", repr(req));
    return err;
  }
};

class AxStore
{
public:
  typedef SpinLock Lock;
  typedef SpinLock::Guard LockGuard;
  struct OpenConfig
  {
    OpenConfig(): log_path_(NULL), ballot_path_(NULL) {}
    ~OpenConfig() {}
    const char* log_path_;
    const char* ballot_path_;
  };
  struct FormatConfig
  {
    FormatConfig(): token_() {}
    ~FormatConfig() {}
    Token token_;
  };
public:
  AxStore(): is_opened_(false) {}
  ~AxStore() { close(); }
public:
  int format(OpenConfig* opencfg, FormatConfig* formatcfg) {
    int err = AX_SUCCESS;
    LockGuard guard(lock_);
    if (NULL == opencfg || NULL == formatcfg)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (is_opened_)
    {
      err = AX_NOT_SUPPORT;
    }
    else if (AX_SUCCESS != (err = ballot_.format(opencfg->ballot_path_, &formatcfg->token_)))
    {
      MLOG(WARN, "ballot format fail: path=%s err=%d", opencfg->ballot_path_, err);
    }
    else if (AX_SUCCESS != (err = log_.format(opencfg->log_path_)))
    {
      MLOG(WARN, "log format fail: path=%s err=%d", opencfg->log_path_, err);
    }
    return err;
  }
  int open(OpenConfig* opencfg) {
    int err = AX_SUCCESS;
    LockGuard guard(lock_);
    if (NULL == opencfg)
    {
      err = AX_INVALID_ARGUMENT;
    }
    else if (is_opened_)
    {
      err = AX_INIT_TWICE;
    }
    else if (AX_SUCCESS != (err = ballot_.open(opencfg->ballot_path_)))
    {
      MLOG(WARN, "ballot open fail: path=%s err=%d", opencfg->ballot_path_, err);
    }
    else if (AX_SUCCESS != (err = log_.open(opencfg->log_path_)))
    {
      MLOG(WARN, "log open fail: path=%s err=%d", opencfg->log_path_, err);
    }
    else
    {
      is_opened_ = true;
    }
    if (AX_SUCCESS != err)
    {
      close();
    }
    return err;
  }
  int close() {
    int err = AX_SUCCESS;
    LockGuard guard(lock_);
    if (is_opened_)
    {
      ballot_.close();
      log_.close();
      is_opened_ = false;
    }
    return err;
  }
  int first_req() {
    int err = AX_SUCCESS;
    LockGuard guard(lock_);
    AppendReq req;
    if (!is_opened_)
    {
      err = AX_NOT_INIT;
    }
    else if (AX_SUCCESS != (err = append(req.make_first_req(ballot_.get_self()))))
    {
      MLOG(WARN, "ballot bootstrap fail: err=%d", err);
    }
    return err;
  }
  int vote(VoteReq& req) {
    int err = AX_SUCCESS;
    Ballot ballot;
    if (AX_SUCCESS != (err = load_lock(&ballot, req.cur_term_)))
    {
      err = AX_FATAL_ERR;
    }
    else
    {
      err = ballot.vote(req.cur_term_, req.max_cursor_, req.candinate_);
      store_unlock(&ballot);
    }
    return err;
  }
  int append(AppendReq& req) {
    int err = AX_SUCCESS;
    Ballot ballot;
    if (AX_SUCCESS != (err = load_lock(&ballot, req.cur_term_)))
    {
      err = AX_FATAL_ERR;
    }
    else
    {
      if (req.start_cursor_.equal(ballot.local_end_cursor_))
      {
        err = AX_CURSOR_NOT_MATCH;
      }
      else if (AX_SUCCESS != (err = log_.append(req)))
      {}
      else
      {
        ballot.update(req.end_cursor_, req.group_config_);
      }
      store_unlock(&ballot);
    }
    return err;
  }
private:
  int load_lock(Ballot* ballot, Term term) {
    int err = AX_SUCCESS;
    if (!lock_.lock())
    {
      err = AX_FATAL_ERR;
    }
    else
    {
      if (AX_SUCCESS != (err = ballot_.load(ballot)))
      {
        MLOG(ERROR, "ballot load fail: err=%d", err);
      }
      else if (AX_SUCCESS != (err = ballot->check_term(term)))
      {
        MLOG(WARN, "check_term fail: err=%d", err);
      }
      if (AX_SUCCESS != err)
      {
        lock_.unlock();
      }
    }
    return err;
  }
  int store_unlock(Ballot* ballot) {
    int err = AX_SUCCESS;
    if (AX_SUCCESS != (err = ballot_.store(ballot)))
    {
      MLOG(ERROR, "ballot store fail: err=%d", err);
    }
    lock_.unlock();
    return err;
  }
private:
  bool is_opened_;
  Lock lock_;
  BallotStore ballot_;
  LogStore log_;
};
#endif /* __OB_AX_LOG_SERVER_H__ */
