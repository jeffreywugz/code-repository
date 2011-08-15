#ifndef _STRLIB_H_
#define _STRLIB_H_
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <fnmatch.h>
#include "core.h"

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

int64_t atox(const char* str)
{
        int64_t i;
        sscanf(str, "%lx", &i);
        return i;
}

char* strcpy2(char* dest, const char* src, char* limit)
{
        while((dest < limit) && (*dest++ = *src++))
                ;
        *--dest = 0;
        return dest;
}

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

bool match(const char* pattern, const char *str)
{
        return !fnmatch(pattern, str, 0);
}

#ifdef TEST
/* TestCase */
int test_hex_dec()
{
        char in[] = "hello,world!";
        char out[64];
        char new_str[64];
        cktrue(_hex_decode_char('F') == 0xF);
        cktrue(_hex_decode_char('A') == 0xA);
        cktrue(_hex_decode_char('a') == 0xa);
        cktrue(_hex_decode_char('0') == 0x0);
        hex_encode_str(out, in);
        hex_decode_str(new_str, out);
        cktrue(strcmp(in, new_str) == 0);
        return 0;
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

int test_strcpy2()
{
        char buf[4096];
        char* p = buf;
        p = strcpy2(p, "hello ", buf+4096);
        p = strcpy2(p, "world!\n", buf+4096);
        printf("%s", buf);
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
        char str1[] = " abc  def\nghi\n  lmn\n ";
        char str2[] = "abc";
        char* p;
        char* tok;
        p = str1;
        while((tok = mystrsep(&p, " \t\n\r")))
                printf("<%s>\n", tok);
        p = str2;
        while((tok = mystrsep(&p, ",")))
                printf("<%s>\n", tok);
        return 0;
}

long test_match()
{
        cktrue(match("*", ""));
        cktrue(match("*", "a"));
        cktrue(match("a*", "a"));
        cktrue(!match("a*", "b"));
        cktrue(match("a*", "ab"));
        cktrue(match("*a", "a"));
        cktrue(match("*a", "ba"));
        cktrue(match("*a", "baa"));
        cktrue(!match("*a", "bac"));
        return 0;
}
#endif

#endif /* _STRLIB_H_ */
