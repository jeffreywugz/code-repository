MPI_Comm comm; MPI_COMM_WORLD
MPI_Datatype datatype; MPI_INT MPI_DOUBLE
MPI_Aint extent;
MPI_STATUS_IGNORE

MPI_Init(&argc,&argv)
MPI_Finalize()

MPI_Comm_size(comm,&size)
MPI_Comm_rank(comm,&rank)
MPI_Comm_dup(comm,&newcomm);
MPI_Comm_split(comm_in, color,  key, &comm_out)

MPI_Send(&buf,count,datatype,dest,tag,comm)
MPI_Recv(&buf,count,datatype,source,tag,comm,&status)

MPI_Barrier(comm)
MPI_Bcast(&buffer,count,datatype,root,comm)
MPI_Scatter(&sendbuf,sendcnt,sendtype,&recvbuf,recvcnt,recvtype,root,comm)
MPI_Gather(&sendbuf,sendcnt,sendtype,&recvbuf,recvcount,recvtype,root,comm)
MPI_Reduce(&sendbuf,&recvbuf,count,datatype,op,root,comm)
MPI_Alltoall(&sendbuf,sendcount,sendtype,&recvbuf,recvcnt,recvtype,comm)

MPI_Type_vector(count,blocklength,stride,oldtype,&newtype)
MPI_Type_struct(count,blocklens[],offsets[],old_types,&newtype)
MPI_Type_extent(datatype,&extent)
MPI_Type_commit(&datatype)
MPI_Type_free(&datatype)
