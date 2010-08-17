#include "util.h"
#include <dlfcn.h>

void tcase_reg(Suite* s, void* func, const char* name)
{
        TCase *tcase = tcase_create(name);
        tcase_add_test(tcase, func);
        suite_add_tcase(s, tcase);
}

void* self_symbol(const char* name)
{
        static void* handle = NULL;
        void* symbol;
        char* error;

        if(handle == NULL){
                handle = dlopen (NULL, RTLD_LAZY);
                if(!handle)panic("%s\n", dlerror());
        }

        dlerror();    /* Clear any existing error */
        symbol = dlsym(handle, name);
        if((error = dlerror()) != NULL)panic("%s\n", error);
        return symbol;
}

