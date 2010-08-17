#include <stdio.h>
#include <string.h>
#include <mpi.h>

void panic(const char* s, int rc)
{
        fprintf(stderr, "%s\n", s);
        MPI_Abort(MPI_COMM_WORLD, rc);
}

int main(int argc, char** argv)
{
        int  np, id, rc;

        rc = MPI_Init(&argc,&argv);
        if(rc!=MPI_SUCCESS)
                panic("init error!", rc);

        MPI_Comm_size(MPI_COMM_WORLD,&np);
        MPI_Comm_rank(MPI_COMM_WORLD,&id);
        printf("np=%d id=%d\n", np, id);
        MPI_Finalize();
        return 0;
}
