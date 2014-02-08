struct Buffer
{
  uint64_t limit_;
  uint64_t used_;
  char* buf_;
};

typedef uint64_t Term;
typedef uint64_t Pos;
struct Cursor
{
  Term term_;
  Pos pos_;
};

struct Timeout
{
  int64_t start_time_;
  int64_t timeout_;
};

struct IOTask
{
  int type_;
  Timeout timeout_;
  Pos pos_;
  Buffer buf_;
  Callback* callback_;
};

class BigFile
{
public:
  int init(const char* path);
  int submit(Task* task);
};

class MetaStore
{
public:
  int init(BigFile* file, Token& init_token); // load meta_ from disk
  int get_term(Term& term);
  int update_term(Term term);
  int get_cur_leader(Term& term, Server& leader);
  int get_my_vote_leader(Term& term, Server& leader);
  int vote_for_leader(Term term, Server& leader);
  int get_group_config(GroupConfig* group_config);
  int update_group_config(Cursor* cursor, GroupConfig* group_config);
  int get_commited_cursor(Cursor* cursor);
  int update_commited_cursor(Cursor* cursor);
protected:
  int do_checkpoint();
private:
  BigFile* file_;
  Lock lock_;
  Meta cur_meta_;
  Cursor commited_cursor_;
  MetaCheckPoint checkpoint_;
};

struct LogIOTask
{
  IOTask io_task_;
  Cursor start_cursor_;
  Cursor end_cursor_;
};

class LogStore
{
public:
  int init(BigFile* file, MetaStore* meta_store);
  int submit(Task* task);
  int catchup();
private:
  int reset(Cursor& cursor);
  int wait();
};

class AxHandler
{
public:
  int main_loop()
  {
    return keep_alive();
  }
  int keep_alive()
  {
    if (AX_SUCCESS != (err = try_init()))
    {}
    else if (AX_SUCCESS != (err = try_recovery()))
    {}
    else if (AX_SUCCESS != (err = try_catchup()))
    {}
  }
};

class StateMgr
{
  enum State {INIT, RECOVERING, ACTIVE};
  State get_state() {}
  int main_loop()
  {
    while(!is_stopped())
    {
    }
  }
  int do_init(){ // load meta
  }
  int do_recovery() { // replay meta
  }
  int do_
};

int do_catchup()
{
  int err = 0;
  return err;
}
           
