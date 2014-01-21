class LogStore
{
};

class LogSync
{
};

class MemberExchange
{
  int get_group();
};

class CommitDecide
{
  int get_commit_cursor(Cursor& cursor);
};

class LeaderElection
{
  int get_leader(Server& server);
};

class AxHandler
{
public:
  int init(Config* config);
  int bootstrap(Server& server);
  int reconfigure();
private:
  LogStore log_store_;
  LogSync log_sync_;
  MemberExchange member_exchange_;
  CommitDecide commit_decide_;
  LeaderElection leader_election_;
};
