#ifndef _SHLIB_H_
#define _SHLIB_H_
#include "container.h"
#include <getopt.h>

#define MAX_N_OPT 128
const char* opt_marker = "just marker.";

typedef struct _opt_spec_t {
        const char* name;
        const char* _default;
        const char* help;
} opt_spec_t;

void _init_opt(struct option* _opt, int n, opt_spec_t* opts)
{
        struct option* p;
        for(p = _opt; p-_opt < n-1 && opts->name; p++, opts++){
                *p = (struct option){opts->name,
                                        opts->_default? required_argument: no_argument,
                                        0, 0};
        }
        *p = (struct option){0, 0, 0, 0};
}

int print_help(const char* usages, opt_spec_t* opts, const char* app_name)
{
        fprintf(stderr, usages, app_name);
        fprintf(stderr, "Option:\n");
        for(opt_spec_t* p = opts; p->name; p++){
                fprintf(stderr, "    --%s%s%s\t%s\n",
                        p->name,
                        p->_default? "=": "",
                        p->_default? p->_default: "",
                        p->help);
        }
        return -1;
}

void print_option_struct(struct option* opt)
{
        for(struct option* p = opt; p->name; p++){
                fprintf(stderr, "struct options{%s, %d, %p, %d};\n",
                        p->name, p->has_arg, p->flag, p->val);
        }

}

int parse_opt(pair_t* li, int n, opt_spec_t* opts, int argc, char** argv)
{
        struct option _opt[MAX_N_OPT];
        
        memset(_opt, 0, sizeof(_opt));
        _init_opt(_opt, MAX_N_OPT, opts);
        for(opt_spec_t* p = opts; p->name; p++){
                assoc_set(li, n, p->name, p->_default);
        }

        optind = 0;
        while(1){
                int index=0;
                char c = getopt_long_only(argc, argv, "", _opt, &index);
                if(c == -1)break;
                if(c != 0)return -1;
                opt_spec_t* p = opts + index;
                assoc_set(li, n, p->name, p->_default? optarg: opt_marker);
        }
        return optind;
}

#ifdef TEST
int test_parse_opt()
{
        opt_spec_t opts[] = {
                {"help", NULL, "this help"},
                {"config", "~/.config", "path of config file"},
                {"port", "2600", "port of server"},
                {0},
        };
        pair_t li[MAX_N_OPT] = {{0,}};
        char* argv[] = {"./a.out", "--help", "--port",  "2700", "start"};
        parse_opt(li, array_len(li), opts, array_len(argv), argv);
        assoc_print(li, array_len(li));
        cktrue(assoc_get(li, array_len(li), "help"));
        return 0;
}
#endif

typedef struct _opt_parser_t {
        opt_spec_t* opts;
        const char* usages;
        pair_t kv[MAX_N_OPT];
        char* app_name;
        int argc;
        char** argv;
} opt_parser_t;

int opt_parser_init(opt_parser_t* self, const char* usages, opt_spec_t* opts, int argc, char** argv)
{
        memset(self->kv, 0, sizeof(self->kv));
        self->app_name = argv[0];
        int rest_idx = parse_opt(self->kv, array_len(self->kv), opts, argc, argv);
        self->opts = opts;
        self->usages = usages;
        self->argv = argv + rest_idx;
        self->argc = argc - rest_idx;
        return rest_idx;
}

char* opt_parser_get(opt_parser_t* self, const char* key)
{
        return (char*)assoc_get(self->kv, array_len(self->kv), key);
}

void opt_parser_print(opt_parser_t* self)
{
        assoc_print(self->kv, array_len(self->kv));
}

int opt_parser_help(opt_parser_t* self)
{
        return print_help(self->usages, self->opts, self->app_name);
}

#ifdef __cplusplus
class CmdOptParser {
public:
        CmdOptParser(const char* usages, opt_spec_t* opts, int argc, char** argv) {
                opt_parser_init(&parser, usages, opts, argc, argv);
        }
        char* get(const char* key){
                return opt_parser_get(&parser, key);
        }
        void print() {
                opt_parser_print(&parser);
        }
        int help() {
                return opt_parser_help(&parser);
        }
        opt_parser_t parser;
};
#endif

#endif /* _SHLIB_H_ */
