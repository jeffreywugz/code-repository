#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct cpio_header {
        char    magic[6];
        char    ino[8];
        char    mode[8];
        char    uid[8];
        char    gid[8];
        char    nlink[8];
        char    mtime[8];
        char    filesize[8];
        char    devmajor[8];
        char    devminor[8];
        char    rdevmajor[8];
        char    rdevminor[8];
        char    namesize[8];
        char    check[8];
} __attribute__((packed));

struct cpio_file
{
        int namesize;
        char *name;
        int filesize;
        char *file;
        unsigned short mode;
};

struct cpio_header* cpio_header_new_()
{
        struct cpio_header *header;
        header=malloc(sizeof(*header));
        return header;
}
        
struct cpio_header* cpio_header_new(FILE* fp)
{
        struct cpio_header *header;
        header=cpio_header_new_();
        fread(header, sizeof(*header), 1, fp);
        return header;
}

int natoi(const char *s, int n)
{
        static char t[20];
        int i;
        strncpy(t, s, n);
        sscanf(t, "%x", &i);
        return i;
}
#define cpio_parse_len(x) natoi(x, sizeof(x))

void cpio_parse_header(struct cpio_file *file, struct cpio_header *header)
{
        file->namesize=cpio_parse_len(header->namesize);
        file->filesize=cpio_parse_len(header->filesize);
        file->mode=cpio_parse_len(header->mode);
}

struct cpio_file* cpio_file_new_(struct cpio_header *header)
{
        struct cpio_file *file;
        file=malloc(sizeof(*file));
        cpio_parse_header(file, header);
        file->name=malloc(file->namesize);
        file->file=malloc(file->filesize);
        return file;
}

long align(long x, long a)
{
        return (x+a-1) & ~(a-1);
}

void cpio_fp_align(FILE *fp)
{
        long pos;
        pos=ftell(fp);
        pos=align(pos, 4);
        fseek(fp, pos, SEEK_SET);
}

void cpio_file_read(struct cpio_file *file, FILE *fp)
{
        fread(file->name, file->namesize, 1, fp);
        cpio_fp_align(fp);
        fread(file->file, file->filesize, 1, fp);
        cpio_fp_align(fp);
}

struct cpio_file* cpio_file_new(FILE *fp)
{
        struct cpio_header *header;
        struct cpio_file *file;
        header=cpio_header_new(fp);
        file=cpio_file_new_(header);
        cpio_file_read(file, fp);
        return file;
}

void cpio_file_print(struct cpio_file *file)
{
        /* printf("[%s]:\n%.*s\n", file->name, file->filesize, file->file); */
        /* printf("-----------------------------------------------\n"); */
        printf("%s:%d:%xs\n", file->name, file->filesize, file->mode);
}

int cpio_end(struct cpio_file *file)
{
        return !strcmp(file->name, "TRAILER!!!");
}

int main()
{
        FILE *fp=stdin;
        struct cpio_file *file;
        while(1){
                file=cpio_file_new(fp);
                if(cpio_end(file))break;
                cpio_file_print(file);
        }
        return 0;
}
