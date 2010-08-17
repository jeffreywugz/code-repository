#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util.h"
#include "config_file.h"


int config_file_init(ConfigFile* config_file, const char* file)
{
        strcpy(config_file->file, file);
        config_file->n_item = 0;
        return 0;
}

void config_file_print(ConfigFile* config_file)
{
        int i;
        printf("config file: %s\n", config_file->file);
        for(i = 0; i<config_file->n_item; i++){
                printf("%s = %s\n",
                       config_file->dict[i][0], config_file->dict[i][1]);
        }
}

static int config_file_buf_read(ConfigFile* config_file)
{
        FILE* fp;
        int file_len;
        fp = fopen(config_file->file, "r");
        if(!fp)panic("can't open file:%s\n", config_file->file);
        file_len = fread(config_file->file_buf, 1, CONFIG_FILE_MAX_LEN, fp);
        fclose(fp);
        config_file->file_buf[file_len] = 0;
        return 0;
}

static int config_file_buf_write(ConfigFile* config_file)
{
        FILE* fp;
        int file_len;
        fp = fopen(config_file->file, "w");
        if(!fp)panic("can't open file:%s\n", config_file->file);
        file_len = strlen(config_file->file_buf);
        fwrite(config_file->file_buf, 1, file_len, fp);
        fclose(fp);
        return 0;
}

static int config_file_buf_update(ConfigFile* config_file)
{
        char line[CONFIG_FILE_MAX_NAME_LEN*2 + 2];
        const char* ikey;
        const char* ival;
        int i;
        config_file->file_buf[0] = '\0';
        for(i = 0; i<config_file->n_item; i++){
                ikey = config_file->dict[i][0];
                ival = config_file->dict[i][1];
                sprintf(line, "%s=%s\n", ikey, ival);
                strcat(config_file->file_buf, line);
        }
        return 0;
}

static int config_file_get_lines(ConfigFile* config_file, char* lines[])
{
        int i;
        char *p;
        config_file_buf_read(config_file);
        p = strtok(config_file->file_buf, "\n");
        if(!p)return 0;
        lines[0] = p;
        for(i = 1; (p = strtok(NULL, "\n")); i++)
                lines[i] = p;
        return i;
}

static int config_file_parse_line(char* line, char** key, char** val)
{
        char *p;
        p = strtok(line, " = ");
        if(!p)return -1;
        *key = p;
        p = strtok(NULL, ";\n");
        if(!p)return -1;
        *val = p;
        return 0;
}

int config_file_read(ConfigFile* config_file)
{
        char *lines[CONFIG_FILE_MAX_ITEMS];
        char *ikey, *ival;
        int i;
        int err;
        config_file->n_item = config_file_get_lines(config_file, lines);
        if(config_file->n_item == 0){
                panic("config_file_get_lines error: %s\n", config_file->file);
                return -1;
        }
        for(i = 0; i<config_file->n_item; i++){
                err = config_file_parse_line(lines[i], &ikey, &ival);
                if(err < 0){
                        panic("parse_line error: %s\n", lines[i]);
                        return err;
                }
                strcpy(config_file->dict[i][0], ikey);
                strcpy(config_file->dict[i][1], ival);
        }
        return 0;
}

int config_file_write(ConfigFile* config_file)
{
        int err;
        err = config_file_buf_update(config_file);
        if(err < 0)return err;
        config_file_buf_write(config_file);
        if(err < 0)return err;
        return 0;
}

char* config_file_get(ConfigFile* config_file, const char* key)
{
        char* ikey;
        char* ival;
        int i;
        for(i = 0; i<config_file->n_item; i++){
                ikey = config_file->dict[i][0];
                ival = config_file->dict[i][1];
                if(!strcmp(ikey, key)){
                        return ival;
                }
        }
        return NULL;
}

int config_file_put(ConfigFile* config_file, const char* key, const char* val)
{
        char* ikey;
        char* ival;
        int i;
        for(i = 0; i<config_file->n_item; i++){
                ikey = config_file->dict[i][0];
                ival = config_file->dict[i][1];
                if(!strcmp(ikey, key)){
                        strcpy(ival, val);
                        return 0;
                }
        }
        strcpy(config_file->dict[i][0], key);
        strcpy(config_file->dict[i][1], val);
        config_file->n_item++;
        return 0;
}
