#include "utils.h"

const char* str_bool(bool b)
{
  return b? "true" : "false";
}

int min(int64_t x, int64_t y)
{
  return x < y? x: y;
}

static void hex_encode_char(char* out, const char c)
{
        static const char* map = "0123456789ABCDEF";
        *out++ = map[c & 0x0f];
        *out++ = map[(c & 0xf0)>>4];
}

char* mystrsep(char** str, const char* delim)
{
        char* start;
        if(*str == NULL)return NULL;
        *str += strspn(*str, delim);
        start = *str;
        *str += strcspn(*str, delim);
        if(**str == 0)*str = NULL;
        else *(*str)++ = 0;
        return *start? start: NULL;
}

char* str_escape(char* out, const char* in, const int64_t len)
{
  int err = OB_SUCCESS;
  char* result = out;
  int64_t i = len;
  if (NULL == out || NULL == in || 0 > len)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "str_escape(out=%p, in=%p, len=%ld)=>%d", out, in, len, err);
  }
  while(OB_SUCCESS == err && i--)
  {
    if(isgraph(*in) && *in != '\\')
    {
      *out++ = *in++;
    }
    else
    {
      *out++ = '\\';
      if(*in == '\\')*out++ = '\\';
      else if(*in == ' ')*out++ = 's';
      else if(*in == '\f')*out++ = 'f';
      else if(*in == '\t')*out++ = 't';
      else if(*in == '\v')*out++ = 'v';
      else if(*in == '\n')*out++ = 'n';
      else if(*in == '\r')*out++ = 'r';
      else *out++ = 'x', hex_encode_char(out, *in), out += 2;
      in++;
    }
  }
  if (OB_SUCCESS == err)
  {
    *out = 0;
  }
  else
  {
    result = NULL;
  }
  return result;
}


const char* obj_type_repr(const ObObjType _type)
{
  const char* type_name[] = {"null", "int", "float", "double", "datetime", "precisedatetime", "varchar", "seq", "createtime", "modifytime", "extendtype"};
    // enum ObObjType
    // {
    //   ObMinType = -1,
    //   ObNullType,   // 空类型
    //   ObIntType,
    //   ObFloatType,
    //   ObDoubleType,
    //   ObDateTimeType,
    //   ObPreciseDateTimeType,
    //   ObVarcharType,
    //   ObSeqType,
    //   ObCreateTimeType,
    //   ObModifyTimeType,
    //   ObExtendType,
    //   ObMaxType,
    // };
  return (_type > ObMinType && _type < ObMaxType) ? type_name[_type]: "unknown";
}
int to_obj(ObObj& obj, const int64_t v)
{
  int err = OB_SUCCESS;
  obj.reset();
  obj.set_int(v);
  return err;
}

int to_obj(ObObj& obj, const ObString& v)
{
  int err = OB_SUCCESS;
  obj.reset();
  obj.set_varchar(v);
  return err;
}

int to_obj(ObObj& obj, const char* v)
{
  int err = OB_SUCCESS;
  ObString _v;
  int64_t max_str_len = 1024;
  _v.assign_ptr((char*)v, v? strnlen(v, max_str_len): 0);
  obj.reset();
  obj.set_varchar(_v);
  return err;
}
    
int alloc_str(char* buf, const int64_t len, int64_t& pos, ObString& str, const char* _str)
{
  int err = OB_SUCCESS;
  int64_t old_pos = pos;
  if (OB_SUCCESS != (err = strformat(buf, len, pos, "%s", _str)))
  {
    TBSYS_LOG(ERROR, "strformat(buf=%p, len=%ld, pos=%ld, _str=%s)=>%d", buf, len, pos, _str, err);
  }
  else
  {
    str.assign_ptr(buf + old_pos, pos - old_pos);
    pos++;
  }
  return err;
}

int alloc_str(char* buf, const int64_t len, int64_t& pos, ObString& str, const ObString _str)
{
  int err = OB_SUCCESS;
  err = OB_NOT_SUPPORTED;
  return err;
}

int vstrformat(char* buf, const int64_t len, int64_t& pos, const char* format, va_list ap)
{
  int err = OB_SUCCESS;
  int64_t count = 0;
  if (NULL == buf || 0 > len || len < pos)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "strformat(buf=%s, len=%ld, pos=%ld, format='%s')=>%d", buf, len, pos, format, err);
  }
  else if (len - pos <= 0)
  {
    err = OB_BUF_NOT_ENOUGH;
    TBSYS_LOG(ERROR, "strformat(buf=%s, len=%ld, pos=%ld, format='%s')=>%d", buf, len, pos, format, err);
  }
  else
  {
    if (0 > (count = vsnprintf(buf + pos, len - pos - 1, format, ap)) || count > len - pos -1)
    {
      err = OB_BUF_NOT_ENOUGH;
      TBSYS_LOG(ERROR, "strformat(buf=%s, len=%ld, pos=%ld, format='%s')=>[count=%ld,err=%d]",
                buf, len, pos, format, count, err);
    }
  }
  
  if (OB_SUCCESS == err)
  {
    pos += count;
  }
  return err;
}

int strformat(char* buf, const int64_t len, int64_t& pos, const char* format, ...)
{
  int err = OB_SUCCESS;
  va_list ap;
  va_start(ap, format);
  err = vstrformat(buf, len, pos, format, ap);
  va_end(ap);
  return err;
}

const char* DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S";
int time_format(char* buf, const int64_t len, int64_t& pos, const int64_t time_us, const char *format)
{
  int err = OB_SUCCESS;
  int64_t count = 0;
  struct tm time_struct;
  int64_t time_s = time_us / 1000000;
  if (NULL == buf || NULL == format)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "time_format(buf=%p, format=%p)=>%d", buf, format, err);
  }
  else if(NULL == localtime_r(&time_s, &time_struct))
  {
    err = OB_ERR_UNEXPECTED;
    TBSYS_LOG(ERROR, "localtime(time_us=%ld)=>NULL", time_us);
  }
  else if (len - pos <(count = strftime(buf + pos, len - pos, format, &time_struct)))
  {
    err = OB_BUF_NOT_ENOUGH;
    TBSYS_LOG(ERROR, "strftime(buf_size=%ld, format='%s')=>%d", len-pos, format, err);
  }
  return err;
}

const char* time2str(const int64_t time_us, const char* format=DEFAULT_TIME_FORMAT);
const char* time2str(const int64_t time_us, const char* format)
{
  static __thread char buf[128];
  int err = OB_SUCCESS;
  int64_t pos = 0;
  if (OB_SUCCESS != (err = time_format(buf, sizeof(buf), pos, time_us, format)))
  {
    TBSYS_LOG(ERROR, "time_format()=>%d", err);
  }
  return OB_SUCCESS == err? buf: NULL;
}

int repr(char* buf, const int64_t len, int64_t& pos, const ObObj& value)
{
  int err = OB_SUCCESS;
  ObObjType _type = value.get_type();
  int64_t int_value = 0;
  ObCreateTime create_time = 0;
  ObModifyTime modify_time = 0;
  ObDateTime datetime = 0;
  ObPreciseDateTime precise_datetime = 0;
  float float_value = 0.0;
  double double_value = 0.0;
  ObString vchar_value;
  //str.clear();
  switch(_type)
  {
    case ObNullType:
      if (OB_SUCCESS != (err = strformat(buf, len, pos, "null")))
      {
        TBSYS_LOG(ERROR, "strformat()=>%d", err);
      }
      break;
    case ObIntType:
      if (OB_SUCCESS != (err = value.get_int(int_value)))
      {
        TBSYS_LOG(ERROR, "value.get_int()=>%d", err);
      }
      else if (OB_SUCCESS != (err = strformat(buf, len, pos, "%ld", int_value)))
      {
        TBSYS_LOG(ERROR, "strformat()=>%d", err);
      }
      break;
    case ObVarcharType:
      if (OB_SUCCESS != (err = value.get_varchar(vchar_value)))
      {
        TBSYS_LOG(ERROR, "value.get_varchar()=>%d", err);
      }
      else if (OB_SUCCESS != (err = strformat(buf, len, pos, "\'%.*s\'", vchar_value.length(), vchar_value.ptr())))
      {
        TBSYS_LOG(ERROR, "strformat()=>%d", err);
      }
      break;
    case ObFloatType:
      if (OB_SUCCESS != (err = value.get_float(float_value)))
      {
        TBSYS_LOG(ERROR, "value.get_float()=>%d", err);
      }
      else if (OB_SUCCESS != (err = strformat(buf, len, pos, "%.4f", float_value)))
      {
        TBSYS_LOG(ERROR, "strformat()=>%d", err);
      }
      break;
    case ObDoubleType:
      if (OB_SUCCESS != (err = value.get_double(double_value)))
      {
        TBSYS_LOG(ERROR, "value.get_float()=>%d", err);
      }
      else if (OB_SUCCESS != (err = strformat(buf, len, pos, "%.4lf", double_value)))
      {
        TBSYS_LOG(ERROR, "strformat()=>%d", err);
      }
      break;
    case ObDateTimeType:
      if (OB_SUCCESS != (err = value.get_datetime(datetime)))
      {
        TBSYS_LOG(ERROR, "value.get_datetime()=>%d", err);
      }
      else if (OB_SUCCESS != (err = strformat(buf, len, pos, "%s", time2str(datetime))))
      {
        TBSYS_LOG(ERROR, "strformat()=>%d", err);
      }
      break;
    case ObPreciseDateTimeType:
      if (OB_SUCCESS != (err = value.get_precise_datetime(precise_datetime)))
      {
        TBSYS_LOG(ERROR, "value.get_precise_datetime()=>%d", err);
      }
      else if (OB_SUCCESS != (err = strformat(buf, len, pos, "%s", time2str(precise_datetime))))
      {
        TBSYS_LOG(ERROR, "strformat()=>%d", err);
      }
      break;
    case ObCreateTimeType:
      if (OB_SUCCESS != (err = value.get_createtime(create_time)))
      {
        TBSYS_LOG(ERROR, "value.get_createtime()=>%d", err);
      }
      else if (OB_SUCCESS != (err = strformat(buf, len, pos, "%s", time2str(create_time))))
      {
        TBSYS_LOG(ERROR, "strformat()=>%d", err);
      }
      break;
    case ObModifyTimeType:
      if (OB_SUCCESS != (err = value.get_modifytime(modify_time)))
      {
        TBSYS_LOG(ERROR, "value.get_modifytime()=>%d", err);
      }
      else if (OB_SUCCESS != (err = strformat(buf, len, pos, "%s", time2str(modify_time))))
      {
        TBSYS_LOG(ERROR, "strformat()=>%d", err);
      }
      break;
    case ObSeqType:
      if (OB_SUCCESS != (err = strformat(buf, len, pos, "[seq]")))
      {
        TBSYS_LOG(ERROR, "strformat()=>%d", err);
      }
      break;
    case ObExtendType:
      if (OB_SUCCESS != (err = value.get_ext(int_value)))
      {
        TBSYS_LOG(ERROR, "value.get_int()=>%d", err);
      }
      else if (OB_SUCCESS != (err = strformat(buf, len, pos, "[%ld]", int_value)))
      {
        TBSYS_LOG(ERROR, "strformat()=>%d", err);
      }
      break;
    default:
      if (OB_SUCCESS != (err = strformat(buf, len, pos, "obj[type=%d]", _type)))
      {
        TBSYS_LOG(ERROR, "strformat()=>%d", err);
      }
      break;
  }
  return err;
}

int repr(char* buf, const int64_t len, int64_t& pos, const char* _str)
{
  int err = OB_SUCCESS;
  if (OB_SUCCESS != (err = strformat(buf, len, pos, "%s", _str)))
  {
    TBSYS_LOG(ERROR, "strformat(_str=%p)=>%d", &_str, err);
  }
  return err;
}
  
int repr(char* buf, const int64_t len, int64_t& pos, const ObString& _str)
{
  int err = OB_SUCCESS;
  if (OB_SUCCESS != (err = strformat(buf, len, pos, "%.*s", _str.length(), _str.ptr())))
  {
    TBSYS_LOG(ERROR, "strformat(_str=%p)=>%d", &_str, err);
  }
  return err;
}
    
int repr(char* buf, const int64_t len, int64_t& pos, const ObScanner& scanner, int64_t row_limit /*=-1*/)
{
  int err = OB_SUCCESS;
  err = strformat(buf, len, pos, "scanner[%p]", &scanner);
  // for(ObScanner::RowIterator it = scanner.row_begin(); OB_SUCCESS == err && it != scanner.row_end(); ++it)
  // {
  //   ObCellInfo* cells = NULL;
  //   int64_t n = 0;
  //   if (0 < row_limit && row_idx > row_limit)
  //   {
  //     break;
  //   }
  //   if (OB_SUCCESS != (err = it.get_row(&cells, &n)))
  //   {
  //     TBSYS_LOG(ERROR, "row_iter->get_row()=>%d", err);
  //   }
  //   for(int i = 0; OB_SUCCESS == err && (row_idx % 10) == 0 && i < n; i++)
  //   {
  //     if (OB_SUCCESS != (err = repr(buf, len, pos, cells[i].column_name_)))
  //     {
  //       TBSYS_LOG(ERROR, "repr()=>%d", err);
  //     }
  //     else if (OB_SUCCESS != (err = repr(buf, len, pos, "\t")))
  //     {
  //       TBSYS_LOG(ERROR, "repr('\t')=>%d", err);
  //     }
  //   }
  //   if (OB_SUCCESS == err && (row_idx % 10) == 0 && OB_SUCCESS != (err =  repr(buf, len, pos, "\n")))
  //   {
  //     TBSYS_LOG(ERROR, "repr('\n')=>%d", err);
  //   }
        
  //   for(int i = 0; OB_SUCCESS == err && i < n; i++)
  //   {
  //     if (OB_SUCCESS != (err = repr(buf, len, pos, cells[i].value_)))
  //     {
  //       TBSYS_LOG(ERROR, "repr()=>%d", err);
  //     }
  //     else if (OB_SUCCESS != (err = repr(buf, len, pos, "\t")))
  //     {
  //       TBSYS_LOG(ERROR, "repr('\t')=>%d", err);
  //     }
  //   }
        
  //   if (OB_SUCCESS == err && OB_SUCCESS != (err =  repr(buf, len, pos, "\n")))
  //   {
  //     TBSYS_LOG(ERROR, "repr('\n')=>%d", err);
  //   }
  //   row_idx++;
  // }
  return err;
}

int to_server(ObServer& server, const char* spec)
{
  int err = OB_SUCCESS;
  char* p = NULL;
  char ip[64] = "";
  int32_t port = 0;
  if (NULL == spec)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "spec == NULL");
  }
  else if (NULL == (p = strchr(spec, ':')))
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "strchr(spec='%s', ':')=>NULL", spec);
  }
  else 
  {
    strncpy(ip, spec, min(p - spec, sizeof(ip)));
    port = atoi(p+1);
  }
  if (OB_SUCCESS != err)
  {}
  else if (0 >= port)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "to_server(spec=%s)=>%d", spec, err);
  }
  else
  {
    TBSYS_LOG(INFO, "to_server(ip=%s, port=%d)=>%d", ip, port, err);
    server.set_ipv4_addr(ip, port);
  }
  
  return err;
}

int split(char* buf, const int64_t len, int64_t& pos, const char* str, const char* delim,
          int max_n_secs, int& n_secs, char** secs)
{
  int err = OB_SUCCESS;
  ObString _str;
  char* saveptr = NULL;
  int i = 0;
  if (NULL == buf || NULL == str || NULL == delim || 0 >= max_n_secs || NULL == secs)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "split(buf='%s', str='%s', delim='%s', max_n_secs='%s', secs=%p)=>%d",
              buf, str, delim, max_n_secs, n_secs, err);
  }
  else if (OB_SUCCESS != (err = alloc_str(buf, len, pos, _str, str)))
  {
    TBSYS_LOG(ERROR, "alloc_str(buf=%p, len=%ld, pos=%ld, str='%s')=>%d", buf, len, pos, str, err);
  }
  else if (NULL == (secs[0] = strtok_r(_str.ptr(), delim, &saveptr)))
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "strtok_r(str='%s')=>NULL", _str.ptr());
  }
  else
  {
    for(i = 1; i < max_n_secs; i++)
    {
      if (NULL == (secs[i] = strtok_r(NULL, delim, &saveptr)))
      {
        break;
      }
    }
  }
  if (OB_SUCCESS == err)
  {
    n_secs = i;
  }
  return err;
}

int strformat(ObDataBuffer& buf, const char* format, ...)
{
  int err = OB_SUCCESS;
  va_list ap;
  va_start(ap, format);
  err = vstrformat(buf.get_data(), buf.get_capacity(), buf.get_position(), format, ap);
  va_end(ap);
  return err;
}

int split(ObDataBuffer& buf, const char* str, const char* delim, const int max_n_secs, int& n_secs, char** const secs)
{
  return split(buf.get_data(), buf.get_capacity(), buf.get_position(), str, delim, max_n_secs, n_secs, secs);
}

int parse_servers(const char* tablet_servers, const int max_n_servers, int& n_servers, ObServer *servers)
{
  int err = OB_SUCCESS;
  char server_spec_buf[MAX_STR_BUF_SIZE];
  ObDataBuffer buf(server_spec_buf, sizeof(server_spec_buf));
  char* server_specs[max_n_servers];
  int server_count = 0;
  if (NULL == tablet_servers || NULL == servers || 0 >= max_n_servers)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "parse_servers(table_server=%s, max_n_servers=%d, servers=%p)=>%d",
              tablet_servers, max_n_servers, servers, err);
  }
  else if ((int)ARRAYSIZEOF(server_specs) > max_n_servers)
  {
    err = OB_BUF_NOT_ENOUGH;
    TBSYS_LOG(ERROR, "ARRAYSIZEOF(server_specs)[%ld] > max_n_servers[%ld]", ARRAYSIZEOF(server_specs), max_n_servers);
  }
  else if (OB_SUCCESS != (err = split(buf, tablet_servers, ", ", max_n_servers, server_count, server_specs)))
  {
    TBSYS_LOG(ERROR, "split(tablet_servers='%s', max_n_servers=%d)=>%d", tablet_servers, max_n_servers, err);
  }
  else
  {
    for(int i = 0; OB_SUCCESS == err && i < server_count; i++)
    {
      if (OB_SUCCESS != (err = to_server(servers[i], server_specs[i])))
      {
        TBSYS_LOG(ERROR, "to_server(spec='%s')=>%d", server_specs[i], err);
      }
    }
  }
  if (OB_SUCCESS == err)
  {
    n_servers = server_count;
  }
  return err;
}

int read_file(const char* path, char* buf, const int64_t buf_size, int64_t& read_bytes)
{
  int err = OB_SUCCESS;
  int fd = -1;
  if (NULL == path || NULL == buf || 0 >= buf_size)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "read_file(path=%s, buf=%p, buf_size=%ld)=>%d", path, buf, buf_size, err);
  }
  else if (0 > (fd = open(path, O_RDONLY)))
  {
    err = OB_ERR_SYS;
    TBSYS_LOG(ERROR, "open(path=%s)=>%d:%s", path, errno, strerror(errno));
  }
  else if (0 > (read_bytes = read(fd, buf, buf_size-1)))
  {
    err = OB_ERR_SYS;
    TBSYS_LOG(ERROR, "read(path=%s)=>%d:%s", path, errno, strerror(errno));
  }
  else
  {
    buf[read_bytes] = 0;
  }
  close(fd);
  return err;
}

int read_line(char* buf, const int64_t buf_size, int64_t& read_bytes, const char* prompt)
{
  int err = OB_SUCCESS;
  if (NULL == buf || 0 >= buf_size || NULL == prompt)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "read_line(buf=%p, buf_size=%ld, prompte=%s)=>%d", buf, buf_size, prompt, err);
  }
  else if (0 > fprintf(stdout, "%s", prompt))
  {
    err = OB_ERR_SYS;
    TBSYS_LOG(ERROR, "fprintf(prompt='%s') < 0", prompt);
  }
  else if (NULL == fgets(buf, buf_size, stdin))
  {
    err = OB_ITER_END;
    TBSYS_LOG(INFO, "fgets()=>NULL");
  }
  return err;
}

int execute_str(Interpreter* interpreter, const char* str, bool keep_going, const char* delim)
{
  int err = OB_SUCCESS;
  int execute_err = OB_SUCCESS;
  static char buf[MAX_STR_BUF_SIZE];
  char* tmp_str = NULL;
  char* statmt = NULL;
  TBSYS_LOG(INFO, "execute_str(interpreter=%p, cmd='%s', keep_going=%s, delim='%s')",
            interpreter, str, str_bool(keep_going), str_escape(buf, delim, strnlen(delim, sizeof(buf))));
  if (NULL == interpreter || NULL == str)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "execute_str(interpreter=%p, str='%s')=>%d", interpreter, str, err);
  }
  else if (NULL == (tmp_str = strdup(str)))
  {
    err = OB_ALLOCATE_MEMORY_FAILED;
    TBSYS_LOG(ERROR, "strdup()=>%d", err);
  }
  while(OB_SUCCESS == err)
  {
    if (NULL == (statmt = mystrsep(&tmp_str, delim)))
    {
      break;
    }
    else if (OB_SUCCESS != (execute_err = interpreter->execute(statmt)))
    {
      if (!keep_going)
      {
        err = execute_err;
      }
      if (OB_PARTIAL_FAILED == execute_err)
      {
        TBSYS_LOG(ERROR, "interpreter->execute(statmt='%s')=>%d[may be syntax error]", statmt, execute_err);
      }
      else
      {
        TBSYS_LOG(ERROR, "interpreter->execute(statmt='%s')=>%d", statmt, execute_err);
      }
    }
  }
  if (NULL != tmp_str)
  {
    free(tmp_str);
  }
  return err;
}

int execute_file(Interpreter* interpreter, const char* path, bool keep_going, const char* delim)
{
  int err = OB_SUCCESS;
  int64_t read_bytes  = 0;
  static char buf[MAX_STR_BUF_SIZE];
  if (NULL == interpreter || NULL == path || NULL == delim)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "execute_str(interpreter=%p, path='%s', delim='%s')=>%d", interpreter, path, delim, err);
  }
  else if (OB_SUCCESS != (err = read_file(path, buf, sizeof(buf), read_bytes)))
  {
    TBSYS_LOG(ERROR, "read_file(path='%s'), buf_size=%ld)=>%d", path, sizeof(buf), err);
  }
  else if (OB_SUCCESS != (err = execute_str(interpreter, buf, keep_going, delim)))
  {
    TBSYS_LOG(ERROR, "execute_str(str='%s')=>%d", buf, err);
  }
  return err;
}

int execute_interactive(Interpreter* interpreter, bool keep_going, const char* delim, const char* prompt)
{
  int err = OB_SUCCESS;
  int execute_err = OB_SUCCESS;
  int64_t read_bytes = 0;
  static char buf[MAX_STR_BUF_SIZE];
  if (NULL == interpreter || NULL == delim || NULL == prompt)
  {
    err = OB_INVALID_ARGUMENT;
    TBSYS_LOG(ERROR, "execute_str(interpreter=%p, delim='%s', prompt='%s')=>%d", interpreter, delim, prompt, err);
  }
  while(OB_SUCCESS == err)
  {
    if (OB_SUCCESS != (err = read_line(buf, sizeof(buf), read_bytes, prompt)))
    {
      if (OB_ITER_END != err)
      {
        TBSYS_LOG(ERROR, "read_line(buf_size=%ld)=>%d", sizeof(buf), err);
      }
      else
      {
        TBSYS_LOG(INFO, "quit.");
        err = OB_SUCCESS;
        break;
      }
    }
    else if (OB_SUCCESS != (execute_err = execute_str(interpreter, buf, keep_going, delim)))
    {
      if (!keep_going)
      {
        err = execute_err;
      }
      TBSYS_LOG(ERROR, "execute_str(str='%s')=>%d", buf, err);
    }
  }
  return err;
}
