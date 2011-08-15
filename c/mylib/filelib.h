#ifndef _FILELIB_H_
#define _FILELIB_H_
#include "core.h"
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

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

#endif /* _FILELIB_H_ */
