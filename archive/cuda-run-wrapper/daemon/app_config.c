#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util.h"
#include "app_config.h"

static int app_config_option_parse(AppConfig* app_cfg, int argc, char** argv)
{
        struct CmdOptionItem* item;
        if(cmd_option_parse(&app_cfg->cmd_opt, argc, argv) < 0){
                cmd_option_help(&app_cfg->cmd_opt);
                return -1;
        }
        item = cmd_option_get(&app_cfg->cmd_opt, "help");
        if(item == NULL)return -1;
        app_cfg->help = item->is_appeared;

        if(app_cfg->help){
                cmd_option_help(&app_cfg->cmd_opt);
                return -2;
        }
        
        item = cmd_option_get(&app_cfg->cmd_opt, "test");
        if(item == NULL)return -1;
        app_cfg->test = item->is_appeared;
        
        item = cmd_option_get(&app_cfg->cmd_opt, "config-file");
        if(item == NULL)return -1;
        app_cfg->config_file = item->is_appeared;
        if(app_cfg->config_file)
                strcpy(app_cfg->file, item->arg);
        
        return 0;
}

int app_config_option_init(AppConfig* app_cfg, int argc, char** argv)
{
        struct CmdOptionItem opt_items[]={
                {name: "test", has_arg: 0, description: "execute self test."},
                {name: "help", has_arg: 0, description: "show this help."},
                {name: "config-file", has_arg: 1},
        };
        int n_items = array_len(opt_items);
        char usage[CMD_OPTION_MAX_DOC_LEN];
        sprintf(usage, "%s --test | --help\n",
                argv[0]);
        cmd_option_init(&app_cfg->cmd_opt, opt_items, n_items, usage);
        return app_config_option_parse(app_cfg, argc, argv);
}

static int app_config_file_parse(AppConfig* app_cfg)
{
        int err;
        const char* val;
        err = config_file_read(&app_cfg->cfg_file);
        if(err < 0){
                panic("config_file_parse error:");
                return err;
        }
        val = config_file_get(&app_cfg->cfg_file, "server.addr");
        if(val != NULL)
                app_cfg->server_addr = val;
        val = config_file_get(&app_cfg->cfg_file, "server.max-queued-requests");
        if(val != NULL)
                app_cfg->server_max_queued_requests = atoi(val);
        val = config_file_get(&app_cfg->cfg_file, "daemon.max-cache-time-interval");
        if(val != NULL)
                app_cfg->daemon_max_cache_time_interval = atoi(val);
        val = config_file_get(&app_cfg->cfg_file, "daemon.max-cache-requests");
        if(val != NULL)
                app_cfg->daemon_max_cache_requests = atoi(val);
        return 0;
}

int app_config_file_init(AppConfig* app_cfg, const char* file)
{
        if(!app_cfg->config_file)
                strcpy(app_cfg->file, file);
        config_file_init(&app_cfg->cfg_file, app_cfg->file);
        return app_config_file_parse(app_cfg);
}

int app_config_init(AppConfig* app_cfg, int argc, char** argv, const char* file)
{
        int err;
        err = app_config_option_init(app_cfg, argc, argv);
        if(err < 0)return err;
        err = app_config_file_init(app_cfg, file);
        if(err < 0)return err;
        return 0;
}

void app_config_print(AppConfig* app_cfg)
{
       cmd_option_print(&app_cfg->cmd_opt);
       config_file_print(&app_cfg->cfg_file);
}

