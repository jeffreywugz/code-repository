#include <stdio.h>
#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC (void*)(0)
#endif

void checkCUDAError(const char *msg);

__device__ void sum_block(float *s, float *sdata)
{
        int blockSize=blockDim.x;
        int tid=threadIdx.x;
        if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
        if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
        if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

#ifndef  __DEVICE_EMULATION__     
        if (tid < 32)
#endif
        {
                if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; EMUSYNC; }
                if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; EMUSYNC; }
                if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; EMUSYNC; }
                if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; EMUSYNC; }
                if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; EMUSYNC; }
                if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; EMUSYNC; }
        }
    
        if (tid == 0) *s = sdata[0];
}

__global__ void sum_global(float* s, float *array)
{
        extern __shared__ float sdata[];
        int tid=threadIdx.x;
        sdata[tid] = array[tid];
        __syncthreads();
        sum_block(s, sdata);
}

__global__ void calc_pi0(float *da_pi)
{
        extern __shared__ float sdata[];
        int i= blockIdx.x*blockDim.x + threadIdx.x;
        int tid=threadIdx.x;
        int n= blockDim.x*gridDim.x;
        float x = (i - 0.5)/n;
        sdata[tid] = 4.0/(1 + x*x);
        __syncthreads();
        sum_block(da_pi+blockIdx.x, sdata);
}

int main( int argc, char** argv) 
{
        float pi=0.0;
        float *d_pi;
        float *da_pi;
        /* note: numBlocks and numThreadsPerBlocks must be power of 2 */
        int numBlocks = 256;
        int numThreadsPerBlock = 256;
        int numThreads = numBlocks*numThreadsPerBlock;

        size_t memSize = numBlocks*sizeof(float);
        size_t sharedMemSize = numThreadsPerBlock*sizeof(float);
        cudaMalloc((void**)&da_pi, memSize);
        cudaMalloc((void**)&d_pi, sizeof(float));

        dim3 dimGrid(numBlocks);
        dim3 dimBlock(numThreadsPerBlock);
        calc_pi0<<<dimGrid, dimBlock, sharedMemSize>>>(da_pi);
        cudaThreadSynchronize();
        sum_global<<<1, dimGrid, sharedMemSize>>>(d_pi, da_pi);
        cudaThreadSynchronize();
        
        checkCUDAError("kernel execution");
        cudaMemcpy(&pi, d_pi, sizeof(float), cudaMemcpyDeviceToHost);
        checkCUDAError("cudaMemcpy");
        cudaFree(da_pi);
        cudaFree(d_pi);
    
        pi/=numThreads;
        printf("pi=%f\n", pi);
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
