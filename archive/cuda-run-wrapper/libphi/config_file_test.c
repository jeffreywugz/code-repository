#include <unistd.h>
#include <check.h>
#include "config_file.h"

START_TEST(config_file_test)
{
        ConfigFile cfg_file;
        const char* file = "config.txt";
        char* val;
        
        config_file_init(&cfg_file, file);
        val = config_file_get(&cfg_file, "CC");
        fail_unless(val == NULL);
        
        config_file_put(&cfg_file, "CC", "gcc");
        config_file_put(&cfg_file, "CFLAGS", "-Wall -g");
        val = config_file_get(&cfg_file, "CC");
        fail_unless(val != NULL && strcmp(val, "gcc") == 0);
        val = config_file_get(&cfg_file, "CFLAGS");
        fail_unless(val != NULL && strcmp(val, "-Wall -g") == 0);

        config_file_write(&cfg_file);
        config_file_read(&cfg_file);
        unlink(file);
        val = config_file_get(&cfg_file, "CC");
        fail_unless(val != NULL && strcmp(val, "gcc") == 0);
        val = config_file_get(&cfg_file, "CFLAGS");
        fail_unless(val != NULL && strcmp(val, "-Wall -g") == 0);
        val = config_file_get(&cfg_file, "LDFLAGS");
        fail_unless(val == NULL);
}END_TEST

void config_file_test_add(Suite* s)
{
        TCase *tc_config_file = tcase_create("config_file");
        tcase_add_test(tc_config_file, config_file_test);
        suite_add_tcase(s, tc_config_file);
}
