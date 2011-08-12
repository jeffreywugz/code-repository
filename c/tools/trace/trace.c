#include <stdio.h>
#include <stdlib.h>

static FILE *g_trace_output;
void trace_init()__attribute__((no_instrument_function, constructor));
void trace_destroy() __attribute__((no_instrument_function, destructor));

void trace_init()
{
  g_trace_output = fopen(getenv("TRACE"), "w");
}

void trace_destroy()
{
  if(g_trace_output)
    fclose(g_trace_output);
}

#define trace_output(...) if(g_trace_output)fprintf(g_trace_output, __VA_ARGS__)

void __cyg_profile_func_enter( void *this, void *callsite)
{
  trace_output("E %p\n", this);
}

void __cyg_profile_func_exit( void *this, void *callsite)
{
  trace_output("X %p\n", this);
}

