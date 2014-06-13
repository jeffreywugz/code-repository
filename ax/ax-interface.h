

class Client
{
  int stop_group(ServerList* cur_servers, ServerList* servers);
  int start_group(ServerList* cur_servers, ServerList* servers);
  int set_master(ServerList* cur_servers, ServerList* new_master);
  int read(ReadTask& task);
  int write(WriteTask& task);
};

class StreamStore
{
  int init(Config* cfg);
  int submit(Task* task);
};


class StreamSync
{
  class Callback
  {
    int get_master(Server& server);
    int get_sync_start_cursor(Cursor& cursor);
    int append();
  };
  int init(Callback* callback);
};

class MetaManager
{
  int get_master(Server& server);
};


class AxServer
{
  struct Config
  {
    char* self_;
  };
  int start(
};
