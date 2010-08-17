#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

void panic(const char* s, int rc)
{
    fprintf(stderr, "%s\n", s);
    MPI_Abort(MPI_COMM_WORLD, rc);
}

double calc_pi(MPI_Comm comm, int num)
{
    int istart, iend;
    int id, np, n, i;
    double h, xi, s, pi;
    MPI_Comm_size(comm,&np);
    MPI_Comm_rank(comm,&id);
    n=num/np;
    h=1.0/num;
        
    if(id==np-1){
        istart = id*n;
        iend = num;
    } else {
        istart = id*n;
        iend = istart+n;
    }

    s=0;
    for(i=istart; i<iend; i++){
        xi = i*h;
        s += 4.0/(1.0+xi*xi);
    }
    MPI_Reduce(&s, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    if(id==0)
        pi *= h;
    return pi;
}

void test_pi(MPI_Comm comm, double eps)
{
    double pi;
    int num, id;
    num=sqrt(4.0/(3.0*eps));
    pi=calc_pi(comm, num);
    MPI_Comm_rank(comm,&id);
    if(id==0)
        printf("eps:%f\n", eps);
    if(id==0)
        printf("pi=%f\n", pi);
        
}

int main(int argc, char** argv)
{
    int  rc;
    MPI_Comm comm;
    double eps=1e-9;

    rc = MPI_Init(&argc,&argv);
    if(rc!=MPI_SUCCESS)
        panic("init error!", rc);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);

    test_pi(comm, eps);
    MPI_Finalize();
    return 0;
}
