#ifndef _FILE_TOKENIZER_H_
#define _FILE_TOKENIZER_H_
#include "filelib.h"

#define TOKEN_BUF_SIZE (1<<20)
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

#ifdef TEST
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
#endif

#ifdef __cpluspls
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

#ifdef TEST
int test_file_tokenizer()
{
        FileTokenizer tok(0);
        while(tok.next()){
                fprintf(stderr, "<%s>\n", tok.top());
                fprintf(stderr, "tok: %s\n", tok.repr());
        }
        return 0;
}
#endif
#endif

#endif /* _FILE_TOKENIZER_H_ */
