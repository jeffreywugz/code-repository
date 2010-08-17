#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "cmd_option.h"
#ifndef _CMD_OPTION_H_
#define _CMD_OPTION_H_
//<<<header


#include <getopt.h>

#define CMD_OPTION_MAX_ARG_LEN 128
#define CMD_OPTION_MAX_OPT_ITEMS 128
#define CMD_OPTION_MAX_DOC_LEN 1024
struct CmdOptionItem
{
        char* name;
        int has_arg;
        int is_appeared;
        char arg[CMD_OPTION_MAX_ARG_LEN];
        char* description;
};

struct CmdOption
{
        struct option long_options[CMD_OPTION_MAX_OPT_ITEMS];
        struct CmdOptionItem opt_items[CMD_OPTION_MAX_OPT_ITEMS];
        const char* rest_argv[CMD_OPTION_MAX_OPT_ITEMS];
        int rest_argc;
        int n_items;
        const char* usage;
};
typedef struct CmdOption CmdOption;



//header>>>
#endif /* _CMD_OPTION_H_*/

//<<<func_list|get_func_list


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
//func_list>>>

//<<<test
#ifndef NOCHECK
#include "util.h"

START_TEST(cmd_option_test)
{
        CmdOption cmd_opt;
        struct CmdOptionItem* item;
        struct CmdOptionItem opt_items[]={
                {name: "test", has_arg: 0, description: "execute self test."},
                {name: "help", has_arg: 0, description: "show this help."},
                {name: "config-file", has_arg: 1},
        };
        int n_items = array_len(opt_items);
        char usage[CMD_OPTION_MAX_DOC_LEN];
        char* argv[] ={"app-name", "--test", "--config-file",
                             "file1,file2", "other-file"};
        int argc = array_len(argv);
        int err;
        
        sprintf(usage, "%s --test | --help\n",
                argv[0]);
        
        cmd_option_init(&cmd_opt, opt_items, n_items, usage);
        err = cmd_option_parse(&cmd_opt, argc, argv);
        fail_unless(err == 0, "cmd_option_parse");
        
        item = cmd_option_get(&cmd_opt, "help");
        fail_unless(item && item->is_appeared == 0);
        
        item = cmd_option_get(&cmd_opt, "test");
        fail_unless(item && item->is_appeared == 1);
        
        item = cmd_option_get(&cmd_opt, "config-file");
        fail_unless(item && item->is_appeared == 1 &&
                strcmp("file1,file2", item->arg) == 0);
        
        fail_unless(cmd_opt.rest_argc == 1 &&
                strcmp("other-file", cmd_opt.rest_argv[0]) == 0);
        
}END_TEST

quick_define_tcase_reg(cmd_option)
#endif /* NOCHECK */
//test>>>
