#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <setjmp.h>

void random_exception(jmp_buf env)
{
        longjmp(env, rand()%8);
}

int try_hard()
{
        int jmpret;
        jmp_buf mark;
        jmpret = setjmp(mark);
        if(jmpret == 0){
                random_exception(mark);
        }
        return jmpret;
}

int main(int argc, char *argv[])
{
        int i;
        time_t t;
        srand(time(&t));
        while(1){
                i = try_hard();
                if(i == 6){
                        printf("lucky\n");
                        break;
                } else {
                        printf("bad luck: %d\n", i);
                }
        }
        return 0;
}
