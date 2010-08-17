#include "kmeans.h"

ConfigFile::ConfigFile(const char* file)
{
        FILE* fp;
        int n;
        strcpy(this->file, file);
        fp=fopen(file, "r");
        if(!fp)panic("can't open file:%s\n", file);
        n=fread(buf, 1, 1024, fp);
        fclose(fp);
        buf[n]=0;
        fillDict();
}

ConfigFile::~ConfigFile()
{
}

void ConfigFile::print()
{
        int i;
        for(i=0; i<n_item; i++){
                printf("%s=%s\n", dict[i][0], dict[i][1]);
        }
}

void ConfigFile::fillDict()
{
        char *lines[256];
        char *ikey, *ival;
        int i;
        n_item=getLines(lines);
        for(i=0; i<n_item; i++){
                parseLine(lines[i], ikey, ival);
                dict[i][0]=ikey;
                dict[i][1]=ival;
        }
}

bool ConfigFile::getVal(const char* key, char* val)
{
        char *ikey, *ival;
        int i;
        for(i=0; i<n_item; i++){
                ikey=dict[i][0];
                ival=dict[i][1];
                if(!strcmp(ikey, key)){
                        strcpy(val, ival);
                        return true;
                }
        }
        return false;
}

int ConfigFile::getLines(char* lines[])
{
        int i;
        char *p;
        p=strtok(buf, "\n");
        if(!p)return 0;
        lines[0]=p;
        for(i=1; (p=strtok(NULL, "\n")); i++)
                lines[i]=p;
        return i;
}

bool ConfigFile::parseLine(char* line, char* &key, char *&val)
{
        char *p;
        p=strtok(line, "=");
        if(!p)return false;
        key=p;
        p=strtok(NULL, ";\n");
        if(!p)return false;
        val=p;
        return true;
}
