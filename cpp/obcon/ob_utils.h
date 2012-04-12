#ifndef __OB_OBCON_UTILS_H__
#define __OB_OBCON_UTILS_H__

#include "common/ob_string.h"
#include "common/ob_object.h"
#include "common/ob_server.h"
#include "common/ob_scanner.h"

using namespace oceanbase::common;
const char* obj_type_repr(const ObObjType _type);
int to_obj(ObObj& obj, const int64_t v);
int to_obj(ObObj& obj, const ObString& v);
int to_obj(ObObj& obj, const char* v);

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

#define reg_parse2(pat, str, buf, ...) (__reg_parse(pat, str, buf.get_data(), buf.get_capacity(), buf.get_position(), ##__VA_ARGS__, NULL) != 0? OB_PARTIAL_FAILED: 0)
#endif /* __OB_OBCON_UTILS_H__ */
