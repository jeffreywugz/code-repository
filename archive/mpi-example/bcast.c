#include <stdio.h>
#include <string.h>
#include <mpi.h>

void panic(const char* s, int rc)
{
    fprintf(stderr, "%s\n", s);
    MPI_Abort(MPI_COMM_WORLD, rc);
}

int _MPI_Bcast(void* buffer, int count, MPI_Datatype datatype,
               int root, MPI_Comm comm)
{
    int  np, id;
    int tag=0;
    int dest, src;
    int i;
        
    MPI_Comm_size(comm,&np);
    MPI_Comm_rank(comm,&id);
    id = (id+np-root)%np;
        
    for(i=1; i<np; i *= 2){
        if(id<i){
            dest = id+i;
            if(dest < np)
                MPI_Send (buffer,count,datatype,(dest+root)%np,tag,comm);
        } else if(id<2*i){
            src = id-i;
            MPI_Recv(buffer,count,datatype,(src+root)%np,tag,comm,MPI_STATUS_IGNORE);
        }
    }
    return 0;
}

void test_bcast(MPI_Comm comm)
{
    int sendbuf[4]={1, 2, 3, 4};
    int recvbuf[20]={4, 3, 2, 1};
    int id, root=2;
    MPI_Comm_rank(comm, &id);
    if(id==root)
        memcpy(recvbuf, sendbuf, sizeof(sendbuf));
    _MPI_Bcast(recvbuf,sizeof(sendbuf)/sizeof(int),MPI_INT,root,comm);
    printf("id=%d: %d %d %d %d\n", id, recvbuf[0], recvbuf[1], recvbuf[2], recvbuf[3]);
}

int main(int argc, char** argv)
{
    int  rc;
    MPI_Comm comm;

    rc = MPI_Init(&argc,&argv);
    if(rc!=MPI_SUCCESS)
        panic("init error!", rc);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);

    test_bcast(comm);
    MPI_Finalize();
    return 0;
}
