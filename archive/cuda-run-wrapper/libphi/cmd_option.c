#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "cmd_option.h"

int cmd_option_init(CmdOption* opt, struct CmdOptionItem* opt_items, int n_items,
                const char* usage)
{
        int i;
        assert(n_items<CMD_OPTION_MAX_OPT_ITEMS);
        memcpy(opt->opt_items, opt_items, n_items*sizeof(struct CmdOptionItem));
        for(i=0; i<n_items; i++){
                opt->long_options[i] = (struct option){
                        opt->opt_items[i].name, opt->opt_items[i].has_arg,
                        0, 0
                };
        }
        opt->long_options[i] = (struct option){0, 0, 0, 0};
        opt->n_items = n_items;
        opt->usage = usage;
        return 0;
}

int cmd_option_parse(CmdOption* opt, int argc, char* argv[])
{
        char c;
        int cmd_option_index=0;
        int i;

        optind = 0;
        while(1){
                c = getopt_long_only(argc, argv, "",
                                     opt->long_options, &cmd_option_index);
                if(c == -1)break;
                if(c != 0)return -1;
                opt->opt_items[cmd_option_index].is_appeared = 1;
                if(optarg)strcpy(opt->opt_items[cmd_option_index].arg, optarg);
        }
        opt->rest_argc = argc - optind;
        if (optind < argc) {
                for(i=0; i<opt->rest_argc; i++)
                        opt->rest_argv[i] = argv[i + optind];
        }
        return 0;
}

struct CmdOptionItem* cmd_option_get(CmdOption* opt, const char* name)
{
        int i;
        struct CmdOptionItem* item;
        for(i=0; i<opt->n_items; i++){
                item = opt->opt_items + i;
                if(strcmp(name, item->name) == 0)
                        return item;
        }
        return NULL;
}

void cmd_option_help(CmdOption* opt)
{
        int i;
        char* name;
        char* description;
        printf(opt->usage);
        for(i=0; i<opt->n_items; i++){
                name = opt->opt_items[i].name;
                description = opt->opt_items[i].description;
                if(description == NULL)continue;
                printf("--%s: %s\n", name, description);
        }
}

void cmd_option_print(CmdOption* opt)
{
        int i;
        struct CmdOptionItem* item;
        printf("cmd options:\n");
        for(i=0; i<opt->n_items; i++){
                item = &opt->opt_items[i];
                printf("--%s: %d, %s\n", item->name,
                       item->is_appeared, item->arg);
        }
        printf("rest args:\n");
        for(i=0; i<opt->rest_argc; i++){
                printf("%s ", opt->rest_argv[i]);
        }
        printf("\n");
}
