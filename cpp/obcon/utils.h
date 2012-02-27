#include "common/ob_string.h"
#include "common/ob_object.h"
#include "common/ob_server.h"
#include "common/ob_scanner.h"

using namespace oceanbase::common;
const int MAX_STR_BUF_SIZE = 1<<12;
const char* obj_type_repr(const ObObjType _type);
int to_obj(ObObj& obj, const int64_t v);
int to_obj(ObObj& obj, const ObString& v);
int to_obj(ObObj& obj, const char* v);

char* str_escape(char* out, const char* in, const int64_t len);

int strformat(char* buf, const int64_t len, int64_t& pos, const char* format, ...)
  __attribute__ ((format (printf, 4, 5)));
int strformat(ObDataBuffer& buf, const char* format, ...)
  __attribute__ ((format (printf, 2, 3)));
int split(char* buf, const int64_t len, int64_t& pos, const char* str, const char* delim,
          int max_n_secs, int& n_secs, const char** secs);
int split(ObDataBuffer& buf, const char* str, const char* delim, const int max_n_secs, int& n_secs, char** const secs);
int repr(char* buf, const int64_t len, int64_t& pos, const char* value);
int repr(char* buf, const int64_t len, int64_t& pos, const ObObj& value);
int repr(char* buf, const int64_t len, int64_t& pos, const  ObString& _str);
int repr(char* buf, const int64_t len, int64_t& pos, const ObScanner& scanner, int64_t row_limit=-1);
int alloc_str(char* buf, const int64_t len, int64_t& pos, ObString& str, const char* _str);
int alloc_str(char* buf, const int64_t len, int64_t& pos, ObString& str, const ObString _str);
int to_server(ObServer& server, const char* spec);
int parse_servers(const char* tablet_servers, const int max_n_servers, int& n_servers, ObServer *servers);

template<typename T>
int repr(ObDataBuffer& buf, T& obj)
{
  return repr(buf.get_data(), buf.get_capacity(), buf.get_position(), obj);
}

template<typename T>
int alloc_str(ObDataBuffer& buf, ObString& str, T value)
{
  return alloc_str(buf.get_data(), buf.get_capacity(), buf.get_position(), str, value);
}

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
