#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>


int main(int argc, char *argv[])
{
        char ch;
        char* argument;
        int i;
        
        struct option long_options[] = {
                {"add", 0, NULL, 'a'},
                {"file", 1, NULL, 'f'}, 
                {"help", 0, NULL, 'h'}, 
                { 0,  0,  0,  0}, 
        };
    
        while((ch = getopt_long(argc, argv, "", long_options, NULL)) != -1){
                switch (ch) {
                case 'a': 
                        printf("--add\n");
                        break;
                case 'f':
                        argument = optarg;
                        printf("--file %s\n", argument);
                        break;
                case 'h':
                default:
                        printf("Usage: %s [-a -f file]\n", argv[0]);
                        exit(-1);
                        break;
                }
        }

        printf("rest argument:\n");
        for(i=optind; i < argc; i++)
                printf("%s\n", argv[optind]);
        return 0;
}
