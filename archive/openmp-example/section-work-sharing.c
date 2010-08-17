#include <stdio.h>
#include <omp.h>

int main()
{
#pragma omp parallel
        {
#pragma omp sections nowait
                {
#pragma omp section
                        printf("tid: %d, do job 1\n", omp_get_thread_num());
#pragma omp section
                        printf("tid: %d, do job 2\n", omp_get_thread_num());
                }
        }
        return 0;
}
