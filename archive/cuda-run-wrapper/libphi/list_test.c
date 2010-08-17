#include <stdio.h>
#include <check.h>
#include "util.h"
#include "list.h"

struct name
{
        struct list_head list;
        const char* name;
};

START_TEST(list_for_each_entry_test)
{
        int i;
        struct name* iter;
        char output[128] = "";
        LIST_HEAD(stu_name);
        struct name stu_name_array[] = {
                {name:"A1"},
                {name:"A2"},
                {name:"A3"},
        };
        
        for(i=0; i<array_len(stu_name_array); i++)
                list_add(&stu_name_array[i].list, &stu_name);
        
        list_for_each_entry(iter, &stu_name, list){
                strcat(output, iter->name);
        }
        fail_unless(strcmp(output, "A3A2A1") == 0, "list_for_each_entry");
}END_TEST

void list_test_add(Suite* s)
{
        TCase *tc_list = tcase_create("list");
        tcase_add_test(tc_list, list_for_each_entry_test);
        suite_add_tcase(s, tc_list);
}
