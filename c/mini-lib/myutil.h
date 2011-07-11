#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <execinfo.h>
#include <sys/mman.h>

#define array_len(a) (sizeof(a)/sizeof(a[0]))
#define swap(x, y) {typeof(x) t; t=x; x=y; y=t;}

enum DebugLevel {PANIC, ERROR, WARNING, INFO};

#define CHECK_DEFAULT_LEVEL WARNING
#define check(level, expr, err, ...)                                    \
        _check(level, expr, err, __FILE__, __LINE__,  ##__VA_ARGS__, NULL)
#define ckerr(expr, ...) check(CHECK_DEFAULT_LEVEL, #expr, expr, ##__VA_ARGS__)
#define cktrue(expr, ...) check(CHECK_DEFAULT_LEVEL, #expr, !(expr), ##__VA_ARGS__)
#define _malloc(n) ({void* p = malloc(n); cktrue(p); p;})
void _check(int level, const char* expr, int err,
            const char* file, int line, ...);
#define show_info 0
#if show_info
#define info(format,...) fprintf(stderr, format, ##__VA_ARGS__)
#else
#define info(format,...)
#endif

void show_stackframe(int start, int num)
{
        void *trace[16];
        char **messages = (char **)NULL;
        int i, trace_size = 0;
        trace_size = backtrace(trace, 16);
        messages = backtrace_symbols(trace, trace_size);
        printf("[bt] Execution path:\n");
        for (i=start; i<trace_size && i<(start+num); ++i){
                printf("[bt] %s\n", messages[i]);
                /* free(messages[i]); */
        }
        free(messages);
}

void _check(int level, const char* expr, int err,
            const char* file, int line, ...)
{
#ifndef  NDEBUG
        const char *msg;
        va_list ap;

        if(err == 0)
                return;
        if(level < INFO)
                fprintf(stderr, "%s:%d: ", file, line);
        va_start(ap, line);
        msg = (const char*)va_arg(ap, char *);
        if (msg == NULL)
                msg = expr;
        vfprintf(stderr, msg, ap);
        fprintf(stderr, "\n");
        va_end(ap);
        if(level < WARNING){
                show_stackframe(2, 5);
                exit(-1);
        }
#endif
}

#define MAX_MSG_LEN 4096
/* char* sf(const char* format, ...) */
/* { */
/*         static char s[MAX_STR_LEN]; */
/*         va_list ap; */
/*         int len; */
/*         va_start(ap, format); */
/*         len = vsnprintf(s, MAX_STR_LEN, format, ap); */
/*         cktrue(len >= 0 && len < MAX_STR_LEN); */
/*         va_end(ap); */
/*         return s; */
/* } */

char* sf(char* start, char* end, const char* format, ...)
{
        va_list ap;
        int len;
        va_start(ap, format);
        len = vsnprintf(start, end-start, format, ap);
        va_end(ap);
        return start+len;
}

#define sfod(start, end, ...) ({char* _new_start = sf(start, end,  ##__VA_ARGS__); cktrue(_new_start < end); _new_start;})

/* class StrStream { */
/* public: */
/* StrStream(int capacity): start(0), end(0){ */
/*                 start = malloc(capacity); */
/*                 end = start + capacity; */
/*         } */
/*         ~StrStream(){ */
/*                 if(start)free(start); */
/*         } */
/*         char* append(const char* format, ...) */
/*         { */
/*                 va_list ap; */
/*                 int len; */
/*                 va_start(ap, format); */
/*                 len = vsnprintf(start, end-start, format, ap); */
/*                 va_end(ap); */
/*                 return start; */
/*         } */
/*         char* start; */
/*         char* end; */
/* }; */

char* strcpy2(char* dest, const char* src, char* limit)
{
        while((dest < limit) && (*dest++ = *src++))
                ;
        *--dest = 0;
        return dest;
}

int test_strcpy2()
{
        char buf[4096];
        char* p = buf;
        p = strcpy2(p, "hello ", buf+4096);
        p = strcpy2(p, "world!\n", buf+4096);
        printf("%s", buf);
        return 0;
}

void hex_encode_char(char* out, char c)
{
        static const char* map = "0123456789ABCDEF";
        *out++ = map[c & 0x0f];
        *out++ = map[(c & 0xf0)>>4];
}

char* hex_encode(char* out, const char* in, int len)
{
        int i;
        
        for(i = 0; i < len; i++){
                hex_encode_char(out, *in);
                in++;
                out += 2;
        }
        *out = 0;
        return out;
}

char* hex_encode_str(char* out, const char* in)
{
        return hex_encode(out, in, strlen(in));
}

char _hex_decode_char(char in)
{
        return in > 'f'? 0: in >= 'a'? 10+in-'a': in > 'F'? 0: (in >= 'A'? 10 + in - 'A': (in > '0'? in - '0': 0));
}

char hex_decode_char(const char* in)
{
        return _hex_decode_char(*(in)) | (_hex_decode_char(*(in+1))<<4);
}

char* hex_decode(char* out, const char* in)
{
        while(*in){
                *out++ = hex_decode_char(in);
                in += 2;
        }
        return out;
}

char* hex_decode_str(char* out, const char* in)
{
        out = hex_decode(out, in);
        *out = 0;
        return out;
}

char* str_escape(char* out, const char* in, int len)
{
        while(len--){
                if(isgraph(*in) && *in != '\\'){
                        *out++ = *in++;
                } else {
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
        *out = 0;
        return out;
}

char* str_unescape(char* out, const char* in)
{
        while(*in){
                if(*in != '\\'){
                        *out++ = *in++;
                        continue;
                }
                in++;
                switch(*in){
                case '\\': *out++ = '\\'; in++; break;
                case 's': *out++ = ' '; in++; break;
                case 'f': *out++ = '\f'; in++; break;
                case 't': *out++ = '\t'; in++; break;
                case 'v': *out++ = '\v'; in++; break;
                case 'n': *out++ = '\n'; in++; break;
                case 'r': *out++ = '\r'; in++; break;
                case 'x': *out++ = hex_decode_char(++in); in += 2; break;
                default: *out++ = '\\', *out++ = *in++;
                }
                
        }
        return out;
        
}

int test_str_escape()
{
        char* str = " abc\tdef\nghi\r\f\v\x31\xaa";
        char buf1[1<<10];
        char buf2[1<<10];
        str_escape(buf1, str, strlen(str));
        *str_unescape(buf2, buf1) = 0;
        fprintf(stderr, "buf1: <%s>\nbuf2: <%s>\n", buf1, buf2);
        cktrue(strcmp(str, buf2) == 0);
        return 0;
}

int64_t atox(const char* str)
{
        int64_t i;
        sscanf(str, "%lx", &i);
        return i;
}

void* file_map(const char* path, size_t* len)
{
        int fd;
        fd = open(path, 0);
        if(fd == -1) return NULL;
        *len = lseek(fd, 0, SEEK_END);
        return mmap(NULL, *len, 0, 0, fd, 0);
}

int read_file(int fd, char* buf, int len)
{
        int remain = len;
        int count;
        while(remain > 0){
                count = read(fd, buf, remain);
                if(count <= 0)break;
                remain -= count;
        }
        return len - remain;
}

int write_file(int fd, char* buf, int len)
{
        int remain = len;
        int count;
        while(remain > 0){
                count = write(fd, buf, remain);
                if(count <= 0)break;
                remain -= count;
        }
        return len - remain;
}

char* _read_file_buffered(int fd, char* start, char* end, int size, char** pos, char** _end)
{
        int read_count = 0;
        cktrue(*pos <= *_end);
        if(*_end - *pos >= size)return *pos;
        if(*_end == end){
                info("*Read File Buffer Switch*\n");
                memmove(start, *pos, *_end - *pos);
                *_end = start + (*_end - *pos);
                *pos = start;
                read_count = read_file(fd, *_end, end-*_end);
                *_end += read_count;
        }
        return *pos;
}

char* read_file_buffered(int fd, char* start, char* end, int size, char** pos, char** _end)
{
        _read_file_buffered(fd, start, end, size, pos, _end);
        return *pos == *_end? NULL: *pos;
}

char* read_file_buffered_str(int fd, char* start, char* end, int size, char** pos, char** _end)
{
        if(!read_file_buffered(fd, start, end, size, pos, _end))return NULL;
        **_end = 0;
        return *pos;
}

char* write_file_buffered(int fd, char* start, char* end, int size, char** _end)
{
        int count;
        if(*_end + size > end){
                info("*Write File Buffer Switch*\n");
                count = write_file(fd, start, *_end - start);
                if(count < *_end - start)return NULL;
                *_end = start;
        }
        return *_end;
}

void write_str(int fd, const char* str)
{
        write(fd, str, strlen(str));
}

void write_line(int fd, const char* str)
{
        write_str(fd, str);
        write(fd, "\n", 1);
}

typedef struct _str_kv_item_t {
        const char* key;
        void* value;
} str_kv_item_t;

void* find_str_kv(str_kv_item_t* li, const char* key)
{
        while(li->key){
                if(!strcmp(key, li->key))break;
                li++;
        }
        return li->key? li->value: NULL;
}

#define TOKEN_BUF_SIZE (1<<12)
#define MAX_SINGLE_TOKEN_SIZE (1<<10)
        
typedef struct _file_tokenizer_t {
        int fd;
        int state;
        char* start;
        char* end;
        int size;
        char* cur_pos;
        char* cur_end;
        char* tok;
} file_tokenizer_t;

// space/new_line follow a newline is significant:
/*
enum GET_TOKEN_STATES {GET_TOKEN_IN_TOKEN, GET_TOKEN_IN_SIG_SPACE, GET_TOKEN_IN_SPACE};
char* get_token(int* state, char** str)
{
#define return_token(STATE) *p = 0; *state = STATE; return *str = start;
        char* p = *str;
        char* start = *str;
        if(*p == 0)return NULL;
        while(*p){
                switch(*state){
                case GET_TOKEN_IN_TOKEN:
                        if(*p == '\n'){
                                *p++ = 0;
                                *state = GET_TOKEN_IN_SIG_SPACE;
                                goto ret;
                        } else if(isspace(*p)){
                                *p++ = 0;
                                *state = GET_TOKEN_IN_SPACE;
                                goto ret;
                        }
                        break;
                case GET_TOKEN_IN_SIG_SPACE:
                        if(!isspace(*p)){
                                *--start = '\n';
                                *(p-1) = 0;
                                *state = GET_TOKEN_IN_TOKEN;
                                goto ret;
                        }
                        break;
                case GET_TOKEN_IN_SPACE:
                        if(*p == '\n'){
                                start = p;
                                *state = GET_TOKEN_IN_SIG_SPACE;
                        } else if(!isspace(*p)){
                                start = p;
                                *state = GET_TOKEN_IN_TOKEN;
                        }
                        break;
                default: ckerr(-1);
                }
                p++;
        }
ret:
        *str = p;
        return start;
}
*/

char* mystrsep(char** str, char* delim)
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

char* get_token(char** str)
{
        char* tok;
        tok = mystrsep(str, " \t\n\r");
        return tok;
}

char* get_token_from_file(int fd, char* start, char* end, int size, char** cur_start, char** cur_end)
{
        read_file_buffered_str(fd, start, end, size, cur_start, cur_end);
        return get_token(cur_start);
}

char* file_tokenizer_next(file_tokenizer_t* tokenizer)
{
        return tokenizer->tok = get_token_from_file(tokenizer->fd, tokenizer->start, tokenizer->end, tokenizer->size, &tokenizer->cur_pos, &tokenizer->cur_end);
}

char* file_tokenizer_current(file_tokenizer_t* tokenizer)
{
        return tokenizer->tok;
}

char* file_tokenizer_repr(file_tokenizer_t* tokenizer)
{
        static char buf[MAX_MSG_LEN];
        sprintf(buf, "Tokenizer: %s . %-.80s...", tokenizer->tok, tokenizer->cur_pos);
        return buf;
}

class ITokenizer {
public:
        ITokenizer(){}
        virtual ~ITokenizer(){}
        virtual char* next() = 0;
        virtual char* top() = 0;
        virtual char* repr() = 0;
};

class FileTokenizer: public ITokenizer {
public:
        FileTokenizer(int fd){
                file_tokenizer = (file_tokenizer_t){fd, 0, buf, buf+TOKEN_BUF_SIZE, MAX_SINGLE_TOKEN_SIZE, buf+TOKEN_BUF_SIZE, buf+TOKEN_BUF_SIZE};
        }
        virtual ~FileTokenizer(){}
        char* next(){
                return file_tokenizer_next(&file_tokenizer);
        }
        char* top(){
                return file_tokenizer_current(&file_tokenizer);
        }
        char* repr(){
                return file_tokenizer_repr(&file_tokenizer);
        }
        char buf[TOKEN_BUF_SIZE+1];
        file_tokenizer_t file_tokenizer;
};

#define next_token(tokenizer) ({char* tok = tokenizer->next(); cktrue(tok, "%s", tokenizer->repr()); tok;})

int test_read_write_buffered()
{
        char buf[1<<10];
        const int ahead_buf_size = 1<<8;
        char* read_start;
        char* read_end;
        char* write_end;
        read_start = read_end = buf + sizeof(buf);
        while(1){
                /* fprintf(stderr, "start: %p, end: %p\n", read_start, read_end); */
                if(!read_file_buffered(0, buf, buf + sizeof(buf), ahead_buf_size, &read_start, &read_end))break;
                write_end = read_end;
                if(!write_file_buffered(1, buf, buf + sizeof(buf), ahead_buf_size, &write_end))break;
                cktrue(read_end != buf + sizeof(buf) || write_end == buf);
                read_start = read_end;
        }
        write_file_buffered(1, buf, buf, ahead_buf_size, &read_start);
        return 0;
}

int  test_read_file_buffered_str()
{
        char buf[1<<10+1];
        const int ahead_buf_size = 1<<8;
        char* start = buf + sizeof(buf) - 1;
        char* end = buf + sizeof(buf) - 1;
        while(read_file_buffered_str(0, buf, buf + sizeof(buf) -1, ahead_buf_size, &start, &end)){
                fprintf(stderr, "<%s>\n", start);
                start++;
        }
        return 0;
}

int test_strsep()
{
        char str[] = " abc  def\nghi\n  lmn";
        char* p = str;
        char* tok;
        while(*(tok = strsep(&p, " \t\n\r")))
                printf("<%s>\n", tok);
        return 0;
}

int test_mystrsep()
{
        char str[] = " abc  def\nghi\n  lmn\n ";
        char* p = str;
        char* tok;
        while((tok = mystrsep(&p, " \t\n\r")))
                printf("<%s>\n", tok);
        return 0;
}

int test_strtok()
{
        char str[] = " abc  def\nghi\n  lmn";
        char* p = str;
        char* tok;
        char* delim = " \t\n\r";
        strtok_r(str, delim, &p);
        while((tok = strtok_r(NULL, delim, &p)))
                printf("<%s>\n", tok);
        return 0;
}

int test_file_tokenizer()
{
        FileTokenizer tok(0);
        while(tok.next()){
                fprintf(stderr, "<%s>\n", tok.top());
                fprintf(stderr, "tok: %s\n", tok.repr());
        }
        return 0;
}

int test_hex_dec()
{
        char in[] = "hello,world!";
        char out[64];
        char new_str[64];
        assert(_hex_decode_char('F') == 0xF);
        assert(_hex_decode_char('A') == 0xA);
        assert(_hex_decode_char('a') == 0xa);
        assert(_hex_decode_char('0') == 0x0);
        hex_encode_str(out, in);
        hex_decode_str(new_str, out);
        assert(strcmp(in, new_str) == 0);
        return 0;
}


