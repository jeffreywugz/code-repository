#include <stdio.h>
#include <unistd.h>
#include <omp.h>

int main()
{
#pragma omp parallel
        {
#pragma omp critical
                {
                        sleep(1);
                }
        }
        printf("done\n");
        return 0;
}
