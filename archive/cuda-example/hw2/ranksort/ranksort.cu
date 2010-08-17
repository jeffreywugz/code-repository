#include <stdio.h>
#include <assert.h>

void checkCUDAError(const char *msg);

__global__ void ranksort(int *d_a, int *d_b)
{
        extern __shared__ float sdata[];
        int tid=threadIdx.x;
        int current=d_a[blockDim.x*blockIdx.x+tid], current_order=0;
        for(int i=0; i<blockIdx.x; i++){
                sdata[tid]=sdata[tid+blockDim.x]=d_a[tid+blockDim.x*i];
                __syncthreads();
                for(int j=0; j<blockDim.x; j++) /* note for duplicate number */
                        if(sdata[tid+j]<=current)current_order++;
                __syncthreads();
        }
        for(int i=blockIdx.x+1; i<gridDim.x; i++){
                sdata[tid]=sdata[tid+blockDim.x]=d_a[tid+blockDim.x*i];
                __syncthreads();
                for(int j=0; j<blockDim.x; j++) /* note for duplicate number */
                        if(sdata[tid+j]<current)current_order++;
                __syncthreads();
        }
        sdata[tid]=sdata[tid+blockDim.x]=current;
        __syncthreads();
        for(int j=0; j<blockDim.x; j++) /* note for duplicate number */
                if(sdata[tid+j]<current || (sdata[tid+j]==current && tid+j>=blockDim.x))
                        current_order++;
        __syncthreads();
        d_b[current_order]=current;
}

void vector_print(int *v, int n)
{
        for(int i=0; i<n; i++)
                printf("%d ", v[i]);
        printf("\n");
}

int main( int argc, char** argv) 
{
    int *h_a;
    int *d_a;
    int *d_b;
    
    int numBlocks = 256;
    int numThreadsPerBlock = 256;
    int numThreads = numBlocks*numThreadsPerBlock;
    
    size_t memSize = numBlocks * numThreadsPerBlock * sizeof(int);
    h_a = (int *) malloc(memSize);
    cudaMalloc((void**)&d_a, memSize);
    cudaMalloc((void**)&d_b, memSize);

    for(int i=0; i<numThreads; i++){
            h_a[i]=numThreads-i;
    }
    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
    size_t sharedMemSize=numThreadsPerBlock*2*sizeof(int);
    cudaMemcpy(d_a, h_a, memSize, cudaMemcpyHostToDevice);
    ranksort<<<dimGrid, dimBlock, sharedMemSize>>>(d_a, d_b);
    cudaThreadSynchronize();
    checkCUDAError("kernel execution");
    
    cudaMemcpy(h_a, d_b, memSize, cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy");
    for(int i = 1; i < numBlocks*numThreadsPerBlock; i++){
            assert(h_a[i] >= h_a[i-1]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }                         
}
