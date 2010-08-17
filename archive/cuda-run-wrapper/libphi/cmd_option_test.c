#include <check.h>
#include "util.h"
#include "cmd_option.h"

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

void cmd_option_test_add(Suite* s)
{
        TCase *tc_cmd_option = tcase_create("cmd_option");
        tcase_add_test(tc_cmd_option, cmd_option_test);
        suite_add_tcase(s, tc_cmd_option);
}
