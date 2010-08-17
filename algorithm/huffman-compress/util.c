#include <stdio.h>
#include <stdlib.h>

void fatal(char* msg, int code)
{
	fprintf(stderr, "%s\n", msg);
	exit(code);
}
