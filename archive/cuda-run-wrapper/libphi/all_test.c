#include <check.h>
#include "all_test.h"

void list_test_add(Suite* s);
void queue_test_add(Suite* s);
void stack_test_add(Suite* s);
void slice_test_add(Suite* s);

void cmd_option_test_add(Suite* s);
void config_file_test_add(Suite* s);

void async_queue_test_add(Suite* s);
void thread_pool_test_add(Suite* s);

void unsocket_test_add(Suite* s);

static Suite* all_suite()
{
        Suite *s = suite_create("all");
        
        list_test_add(s);
        queue_test_add(s);
        stack_test_add(s);
        slice_test_add(s);

        cmd_option_test_add(s);
        config_file_test_add(s);

        async_queue_test_add(s);
        thread_pool_test_add(s);
        
        unsocket_test_add(s);
        return s;
}

int all_test()
{
        int number_failed;
        Suite *s = all_suite();
        SRunner *sr = srunner_create (s);
        srunner_run_all(sr, CK_NORMAL);
        number_failed = srunner_ntests_failed(sr);
        srunner_free(sr);
        return number_failed;
}

#ifdef TEST_ANS42LIB
int main(int argc, char *argv[])
{
        return all_test();
}
#endif
