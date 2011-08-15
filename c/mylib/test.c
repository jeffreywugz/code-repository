#include <stdio.h>
#define TEST 1
#include "myutil.h"

char* g_usages = "Usages:\n"
        "    %1$s --help\n"
        "    %1$s --test\n"
        "    %1$s file1 file2\n";
opt_spec_t g_opts[] = {
        {"help", NULL, "this help"},
        {"test", "None", "unit test"},
        {0},
};

#define TestEntry(tcase) {#tcase, (void*)test_ ## tcase}
long self_test(const char* test_cases)
{
        pair_t cases[] = {
                TestEntry(queue),
                TestEntry(stack),
                TestEntry(dict),
                TestEntry(am),
                TestEntry(thread_pool),
        };
        fprintf(stderr, "test_cases: %s\n", test_cases);
        call_by_patterns(cases, array_len(cases), test_cases, NULL, NULL);
        return 0;
}

void sig_handle(int sig)
{
        switch(sig){
        case SIGTERM:
        case SIGINT:
                fprintf(stderr, "catch SIGTERM/SIGINT, exit...\n");
                exit(-1);
                break;
        default:
                break;
        }
}

int main(int argc, char** argv)
{
        char* test_case;
        opt_parser_t parser;
        
        if(opt_parser_init(&parser, g_usages, g_opts, argc, argv) < 0){
                opt_parser_help(&parser);
                exit(-EINVAL);
        }
        sigaction(SIGTERM, sigaction_with_handler(sig_handle), NULL);
        sigaction(SIGINT, sigaction_with_handler(sig_handle), NULL);
        test_case = opt_parser_get(&parser, "test");
        if(strcmp(test_case, "None")){
                return self_test(test_case);
        } else {
                fprintf(stderr, "do work here:\n");
                for(int i = 0; i < parser.argc; i++)
                        fprintf(stderr, "handle %s\n", parser.argv[i]);
                fprintf(stderr, "wait 10s for your interupt\n");
                sleep(10);
        }
        return 0;
}
