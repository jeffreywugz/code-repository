#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
        printf("USER=%s\nHOME=%s\nPATH=%s\n",
               getenv("USER"), getenv("HOME"), getenv("PATH"));
        return 0;
}

