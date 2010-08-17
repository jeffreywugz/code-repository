#include <stdio.h>
#include <omp.h>

int main()
{

        int tid, i, chunk = 2;
        int n = 10;

#pragma omp parallel shared(chunk) private(i, tid)
        {
#pragma omp for schedule(static, chunk) nowait
                for (i=0; i < n; i++){
                        tid = omp_get_thread_num();
                        printf("tid: %d, iter: %d\n", tid, i);
                }
        }
        return 0;
}

