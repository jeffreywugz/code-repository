#include <stdio.h>
#include <unistd.h>

void hello()
{
        printf("hello\n");
}

int main(int argc, char** argv)
{
        for(int i = 0; i < 1<<10; i++){
                hello();
                sleep(1);
        }
        return 0;
}
