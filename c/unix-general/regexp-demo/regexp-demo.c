#define MAX_ERR_LEN 80
#define MAX_LINE_LEN 600

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>
#include <regex.h>

void panic(const char* msg)
{
        fprintf(stderr, "%s\n", msg);
        exit(EXIT_FAILURE);
}

void re_panic(int err, regex_t* re, const char* msg)
{
        static char err_msg[MAX_ERR_LEN];
        regerror(err, re, err_msg, MAX_ERR_LEN);
        fprintf(stderr, "%s: %s\n", msg, err_msg);
        exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
        regex_t re;
        int err;
        char line[MAX_LINE_LEN];
    
        if(argc != 2)panic("usage: regexp-demo 'RE' <file");
    
        if((err = regcomp(&re, argv[1], REG_EXTENDED)) != 0)
                re_panic(err, &re, "regcomp");
    
        while(fgets(line, MAX_LINE_LEN, stdin) != NULL){
                *(index(line, '\n')) = '\0';
                if((err = regexec(&re, line, 0, NULL, 0)) == 0)
                        puts(line);
                else if (err != REG_NOMATCH)
                        re_panic(err, &re, "regexec");
        } 

        regfree(&re);
        return 0;
}
