#ifndef __OB_AX_LOG_SERVER_H__
#define __OB_AX_LOG_SERVER_H__
#include "common.h"

#define MAX_SERVER_COUNT 8
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
  Term term_;
  Pos pos_;
};

struct Token
{
  Server server_;
  uint64_t start_time_;
};

#define MAX_TOKEN_COUNT 8
#define MAX_SERVER_COUNT 16
struct TokenList
{
  TokenList(): count_(0) {}
  ~TokenList() {}
  int count_;
  Token tokens_[MAX_TOKEN_COUNT];
};

struct ServerRegistry
{
  ServerRegistry() {}
  ~ServerRegistry() {}
  void set(Token& token, Cursor& cursor) {
    token_ = token;
    regist_cursor_ = cursor;
  }
  Token token_;
  Cursor regist_cursor_;
};

struct GroupConfig
{
  GroupConfig(): group0_mask_(0), group1_mask_(0) {}
  ~GroupConfig() {}
  GroupConfig& make_first_group(Token& token, Cursor& cursor)
  {
    group0_mask_ = 0x1;
    group1_mask_ = 0;
    servers_[0].set(token, cursor);
    return group_config;
  }
  uint32_t group0_mask_;
  uint32_t group1_mask_;
  ServerRegistry servers_[MAX__COUNT];
};

struct VoteReq
{
  Term cur_term_;
  Cursor max_cursor_;
  Token candinate_;
};

struct LogEntry
{
  RecordHeader header_;
  Term term_;
  Pos pos_;
};

struct AppendReq
{
  AppendReq& make_first_req(Token& token) {
    set_cursor(start_cursor_, 0, 0);
    set_cursor(end_cursor_, 1, 1);
    set_cursor(commit_cursor_, 1, 1);
    group_config_.make_first_group(token, commit_cursor_);
    return *this;
  }
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
      cur_term = term;
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
  int vote(Term term, Cursor end_cursor, Token candinate) {
    int err = AX_SUCCESS;
    if (term != cur_term_)
    {
      err = AX_TERM_NOT_MATCH;
    }
    else if (local_end_ > end_cursor)
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
  Term cur_term_;
  Token self_;
  Token leader_;
  Cursor local_end_cursor_;
  Cursor commit_cursor_;
  GroupConfig group_config_;
};

class BallotStore
{
public:
  BallotStore() {}
  ~BallotStore() {}
public:
  int format(const char* path, Token* token) {
    int err = AX_SUCCESS;
    MLOG(INFO, "ballot format: path=%s token=%s", path, repr(*token));
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
  int load_lock(Ballot* ballot) {
    int err = AX_SUCCESS;
    *ballot = ballot_;
    return err;
  }
  int store_unlock(Ballot* ballot) {
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
  int format(const char* path, int64_t len) {
    int err = AX_SUCCESS;
    MLOG(INFO, "log_store format: path=%s len=%ld", path, len);
    return err;
  }
  int open(const char* path, Ballot* ballot) {
    int err = AX_SUCCESS;
    MLOG(INFO, "log_store open: path=%s len=%ld ballot=%s", path, len, ballot);
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
    FormatConfig(): log_len_(0), token_() {}
    ~FormatConfig() {}
    int64_t log_len_;
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
      err = AX_NOT_SUPPORTED;
    }
    else if (AX_SUCCESS != (err = ballot_.format(opencfg.ballot_path_, formatcfg.token_)))
    {
      MLOG(WARN, "ballot format fail: path=%s err=%d", opencfg->ballot_path_, err);
    }
    else if (AX_SUCCESS != (err = log_.format(opencfg.log_path_, formatcfg.log_len_)))
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
    else if (AX_SUCCESS != (err = log_.open(opencfg->log_path_, ballot_)))
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
    if (AX_SUCCESS != (err = load_lock(&ballot, req.term_)))
    {
      err = AX_FATAL_ERR;
    }
    else
    {
      err = ballot.vote(req.term_, req.max_cursor_, req.candinate_);
      store_unlock(&ballot);
    }
    return err;
  }
  int append(AppendReq& req) {
    int err = AX_SUCCESS;
    Ballot ballot;
    if (AX_SUCCESS != (err = load_lock(&ballot, req.term_)))
    {
      err = AX_FATAL_ERR;
    }
    else
    {
      if (req.start_cursor_ != ballot.local_end_)
      {
        err = AX_CURSOR_NOT_MATCH;
      }
      else if (AX_SUCCESS != (err = log_.append(req)))
      {}
      else
      {
        ballot.update(req.end_cursor_, req.get_group_config());
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
      else if (AX_SUCCESS != (err = ballot_.check_term(term)))
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
