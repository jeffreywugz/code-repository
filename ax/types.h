typedef uint64_t Id;
#define INVALID_ID (~0UL)
// basic struct define
struct Buffer
{
  Buffer(): limit_(0), used_(0), buf_(NULL) {}
  ~Buffer() {}
  int parse(const char* spec);
  void dump();
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
  Server(): ip_(0), port_(0) {}
  ~Server() {}
  int parse(const char* spec);
  uint32_t ip_;
  uint32_t port_;
};

#define MAX_SERVER_COUNT 8
struct ServerList
{
  ServerList(): count_(0) {}
  ~ServerList() {}
  int parse(const char* spec);
  int count_;
  Server servers_[MAX_SERVER_COUNT];
};

typedef uint64_t Term;
typedef uint64_t Pos;
struct Cursor
{
  Cursor(): term_(), pos_() {}
  ~Cursor() {}
  int parse(const char* spec);
  Term term_;
  Pos pos_;
};

struct Token
{
  Server server_;
  uint64_t start_time_;
};

#define MAX_TOKEN_COUNT 8
struct TokenList
{
  TokenList(): count_(0) {}
  ~TokenList() {}
  int count_;
  Token tokens_[MAX_TOKEN_COUNT];
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

