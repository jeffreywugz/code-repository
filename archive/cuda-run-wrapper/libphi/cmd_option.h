#ifndef _CMD_OPTION_H_
#define _CMD_OPTION_H_

#include <getopt.h>
/**
 * @file   cmd_option.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Mon Jul 20 10:03:58 2009
 * 
 * @brief  define class CmdOption which can parse cmd option.
 * 
 * @ingroup libphi
 * 
 */

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

int cmd_option_init(CmdOption* opt, struct CmdOptionItem* opt_items, int n_items,
                    const char* usage);
int cmd_option_parse(CmdOption* opt, int argc, char* argv[]);
struct CmdOptionItem* cmd_option_get(CmdOption* opt, const char* name);
void cmd_option_print(CmdOption* opt);
void cmd_option_help(CmdOption* opt);


#endif /* _CMD_OPTION_H_ */
