/** Interface:
 * init: config -> err
 * start: void -> err
 * stop: void -> err
 * ping: void -> void
 * set_log_level: log_level -> void
 *
 * get_meta void -> meta
 * update_meta meta -> err
 *
 * bootstrap: void -> err
 * start_group server_list -> err
 * stop_group server_list -> err
 *
 * propose cursor buffer -> err
 * read cursor -> err commited_cursor buffer
 */
class IAxHandler
{
public:
  int init(Config* config);
  int start();
  int stop();
  int ping();
  void set_log_level(int log_level);

  int get_leader(Server* server);
  int change_leader(Server* server);
  int get_server_list(ServerList* group);
  int set_server_list(ServerList* group);
  int get_commit_cursor(Cursor* cursor);
  int get_recyled_cursor(Cursor* cursor);
  int set_recyled_cursor(Cursor* cursor);

  int bootstrap();
  int start_group(ServerList* group);
  int stop_group(ServerList* group);

  int propose(const Cursor* cursor, const Buffer* buffer, ProposeCallback* callback);
  int read(const Cursor* cursor, Buffer* buffer, ReadCallback* callback);
};

// basic struct define
struct Buffer
{
  uint64_t limit_;
  uint64_t used_;
  char* buf_;
};

struct RecordHeader
{
  uint32_t magic_;
  uint32_t len_;
  uint64_t checksum_;
};

struct Server
{
  uint32_t ip_;
  uint32_t port_;
};

typedef uint64_t Term;
typedef uint64_t Pos;
struct Cursor
{
  Term term_;
  Pos pos_;
};

struct Token
{
  Server server_;
  uint64_t start_time_;
};

struct GroupConfig
{
  TokenList active_tokens_;
  ServerList group0_;
  ServerList group1_;
};

struct MetaCheckPoint
{
  Cursor frozen_cursor_;
  GroupConfig group_config_;
};

struct MetaRecord
{
  RecordHeader header_;
  uint64_t version_;
  Token self_;
  Term term_;
  Server leader_;
  Cursor commit_cursor_;
  GroupConfig group_config_;
};

struct LogEntry
{
  RecordHeader header_;
  Term term_;
  Pos pos_;
};



