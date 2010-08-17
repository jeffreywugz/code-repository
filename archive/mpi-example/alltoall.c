#include <stdio.h>
#include <string.h>
#include <mpi.h>

void panic(const char* s, int rc)
{
    fprintf(stderr, "%s\n", s);
    MPI_Abort(MPI_COMM_WORLD, rc);
}

/* use scatter */
int _MPI_Alltoall(void* sendbuf,int sendcount,MPI_Datatype sendtype,
                  void* recvbuf,int recvcount,MPI_Datatype recvtype,
                  MPI_Comm comm)
{
    int  np, id;
    int i;
    MPI_Aint extent;
        
    MPI_Comm_size(comm,&np);
    MPI_Comm_rank(comm,&id);
    MPI_Type_extent(recvtype, &extent);
    for(i=0; i<np; i++){
        MPI_Scatter(sendbuf,sendcount,sendtype,(char*)recvbuf+i*extent, 
                    recvcount,recvtype,i,comm) ;
    }
    return 0;
}

/* use send & recv */
int __MPI_Alltoall(void* sendbuf,int sendcount,MPI_Datatype sendtype,
                   void* recvbuf,int recvcount,MPI_Datatype recvtype,
                   MPI_Comm comm)
{
    int  np, id;
    int tag=0;
    int dest, src;
    void *tsendbuf, *trecvbuf;
    int i;
    MPI_Aint sendextent, recvextent;
        
    MPI_Comm_size(comm,&np);
    MPI_Comm_rank(comm,&id);
    MPI_Type_extent(sendtype, &sendextent);
    MPI_Type_extent(recvtype, &recvextent);

    tsendbuf = (char*)sendbuf + sendcount*sendextent*id;
    trecvbuf = (char*)recvbuf + recvcount*recvextent*id;
    memcpy(trecvbuf, tsendbuf, sendcount*sendextent);
    for(i=1; i<np; i++){
        tsendbuf = (char*)sendbuf + sendcount*sendextent*i;
        dest = (id+i)%np;
        MPI_Send(tsendbuf,sendcount,sendtype,dest,tag,comm);
                
        trecvbuf = (char*)recvbuf + recvcount*recvextent*((id+np-i)%np);
        src = (id+np-i)%np;
        MPI_Recv(trecvbuf,recvcount,recvtype,src,tag,comm,MPI_STATUS_IGNORE);
    }
    return 0;
}

void test_alltoall(MPI_Comm comm)
{
    int np, id, i;
    int sendbuf[20];
    int recvbuf[20];
    MPI_Comm_size(comm,&np);
    MPI_Comm_rank(comm, &id);
    for(i=0; i<np; i++)
        sendbuf[i]=id;
    __MPI_Alltoall(sendbuf,1,MPI_INT,recvbuf, 1,MPI_INT,comm);
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

    test_alltoall(comm);
    MPI_Finalize();
    return 0;
}
