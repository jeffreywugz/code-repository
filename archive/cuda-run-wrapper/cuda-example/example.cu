#include <stdio.h>
#include <unistd.h>
#include "cuda.h"

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err){
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }                         
}

__global__ void myFirstKernel(int *d_a  )
{
        int i= blockIdx.x;
        int j=threadIdx.x;
        d_a[i * blockDim.x + j] += 1000 * i + j;
}

int main(int argc, char** argv) 
{
    int *h_a;
    int *d_a;
    int numBlocks = 256;
    int numThreadsPerBlock = 256;
    size_t memSize = numBlocks * numThreadsPerBlock * sizeof(int)* 64;
    int device;
    
    cudaGetDevice(&device);
    printf("enter cuda program\n");
    printf("device: %d\n", device);
    checkCUDAError("cudaSetDevice");
    
    h_a = (int *)malloc(memSize);
    cudaMalloc((void**)&d_a, memSize);
    checkCUDAError("cudaMalloc");
    cudaMemcpy(d_a, h_a, memSize, cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy");
    
    sleep(1);
    myFirstKernel<<<numBlocks,  numThreadsPerBlock>>>(d_a);
    checkCUDAError("kernel execution");

    sleep(1);
    cudaMemcpy(h_a, d_a, memSize, cudaMemcpyDeviceToHost); 
    checkCUDAError("cudaMemcpy");
    
    cudaFree(d_a);
    free(h_a);
    return 0;
}
