struct Buffer
{
  int64_t limit_;
  int64_t used_;
  char* buf_;
};

typedef uint64_t id_t;
struct FileTask
{
  int32_t type_;
  id_t id_;
  size_t pos_;
  Buffer buf_;
};

class BigFile
{
  public:
    int init(const char* path, size_t file_size);
    int submit(Task* task);
    int wait(Task* task);
};

struct Cursor
{
  uint64_t term_;
  uint64_t pos_;
};

struct AppendTask
{
  Cursor prev_cursor_;
  Cursor end_cursor_;
  FileTask io_task_;
};

class Logstore
{
  public:
    int init(BigFile* file, Cursor* recyle_cursor, Cursor* frozen_cursor);
    int append(AppendTask* task);
    int read(ReadTask* task);
};

struct GroupConfig
{
  Term term_;
  TokenList active_tokens_;
  serverList group0_;
  ServerList group1_;
};

struct MainRecord
{
  RecordHeader header_;
  uint64_t version_;
  Server self_;
  Token token_;
  Term term_;
  Server master_;
  Cursor recyle_cursor_;
  Cursor frozen_cursor_;
};

struct LogRecord
{
  RecordHeader header_;
  GroupConfig group_config_;
  char buf_[];
};
