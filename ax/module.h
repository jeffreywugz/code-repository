class LogStore
{
};

class LogSync
{
};

class MetaExchange
{
  int get_group();
};

class CommitGuard
{
  int get_commit_cursor(Cursor& cursor);
};

class LeaderKeeper
{
  int get_leader(Server& server);
};

