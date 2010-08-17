#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <check.h>

static void setup()
{
        printf("setup...\n");
}

static void teardown()
{
        printf("teardown...\n");
}

#define MAX_MSG_LEN 256
START_TEST(demo_test)
{
        fail_unless(1 == 0, "must fail");
        fail_unless(1, "must pass");
}END_TEST

static Suite* demo_suite()
{
        Suite *s = suite_create("demo");
        TCase *tc_core = tcase_create("Core");
        tcase_add_checked_fixture(tc_core, setup, teardown);
        tcase_add_test(tc_core, demo_test);
        suite_add_tcase(s, tc_core);
        return s;
}

int all_test()
{
        int number_failed;
        Suite *s = demo_suite();
        SRunner *sr = srunner_create (s);
        srunner_run_all(sr, CK_NORMAL);
        number_failed = srunner_ntests_failed(sr);
        srunner_free(sr);
        return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char** argv)
{
        if(argc==2 && strcmp(argv[1], "--test")==0)
                return all_test();
        printf("normal run...\n");
        return 0;
}
