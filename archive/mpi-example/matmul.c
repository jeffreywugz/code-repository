#include <stdio.h>
#include <string.h>
#include <mpi.h>

#define M 2
#define N 4
#define NP 2

void panic(const char* s, int rc)
{
        fprintf(stderr, "%s\n", s);
        MPI_Abort(MPI_COMM_WORLD, rc);
}

void matsubmul(int a[][N], int b[][N], int c[][M*NP], int stride)
{
    int i, j, k;
    int id;
    for(i=0; i<M; i++){
        for(j=0; j<M; j++){
            for(k=0; k<N; k++){
                c[i][stride*M + j] += a[i][k]*b[j][k];
            }
        }
    }
}

void MPI_Matmul(int a[][N], int b[][N], int c[][M*NP], MPI_Comm *comm)
{
        int np, id, i, tag=1;
        int next, prev;
        MPI_Comm_size(comm,&np);
        MPI_Comm_rank(comm, &id);
        
        for(i=0; i<np; i++){
            matsubmul(a, b, c, (i+id)%np);
            if(i != np){
                next = (id+1)%np;
                prev = (id+np-1)%np;
                MPI_Send (b, M*N, MPI_INT, next, tag, comm);
                MPI_Recv (b, M*N, MPI_INT, prev, tag, comm,
                          MPI_STATUS_IGNORE);
            }
        }
}

void test_matmul(MPI_Comm comm)
{
        int np, id, i, j;
        int a[M][N];
        int b[M][N];
        int c[M][M*NP];
        
        MPI_Comm_size(comm,&np);
        MPI_Comm_rank(comm, &id);
        
        for(i=0; i<M; i++)
            for(j=0; j<N; j++){
                a[i][j]=id+1;
                b[i][j]=id+1;
            }
        memset(c, 0, sizeof(c));
        
        MPI_Matmul(a, b, c, comm);
        
        printf("id=%d\n", id);
        for(i=0; i<M; i++){
            for(j=0; j<M*NP; j++){
                printf("%2d ", c[i][j]);
            }
            printf("\n");
        }
}

int main(int argc, char** argv)
{
        int  rc;
        MPI_Comm comm;

        rc = MPI_Init(&argc,&argv);
        if(rc!=MPI_SUCCESS)
                panic("init error!", rc);
        MPI_Comm_dup(MPI_COMM_WORLD, &comm);

        test_matmul(comm);
        MPI_Finalize();
        return 0;
}
