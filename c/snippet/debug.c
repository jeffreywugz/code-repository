#include <pthread.h>

void* debug_thread_handle(void* arg)
{
        while(1)pause();
}

__attribute__((no_instrument_function, constructor)) void debug_thread_init()
{
        pthread_t thread;
        pthread_create(&thread, NULL, debug_thread_handle, NULL);
}

