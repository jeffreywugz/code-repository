#include "util.h"

static Suite* suite_new(int n, char* tcase[])
{
        int i;
        Suite *s = suite_create("all");
        void (*tcase_reg)(Suite*);
        for(i=0; i<n; i++){
                tcase_reg = self_symbol(tcase[i]);
                if(tcase_reg == NULL)
                        panic("no test case: %s defined!\n", tcase[i]);
                tcase_reg(s);
        }
        return s;
}

int tcase_run(int n, char* tcase[])
{
        int number_failed;
        Suite *s = suite_new(n, tcase);
        SRunner *sr = srunner_create (s);
        srunner_run_all(sr, CK_NORMAL);
        number_failed = srunner_ntests_failed(sr);
        srunner_free(sr);
        return number_failed;
}

#ifdef TEST_ANS42LIB
int main(int argc, char* argv[])
{
        return tcase_run(argc-1, argv+1);
}
#endif

