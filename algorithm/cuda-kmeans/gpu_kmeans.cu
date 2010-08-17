#include <math.h>
#include "kmeans.h"

#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC (void*)(0)
#endif

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }                         
}

static void vec_print(float *d_vec, int len)
{
    float vec[128];
    cudaMemcpy(vec, d_vec, len*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0; i<len; i++)
            printf("%f ", vec[i]);
    printf("\n");
}

__device__ void load_center(float *target, float* source, int len)
{
        for( int i = 0; i < len+1; i++ )
                target[i] = source[i];
}

__device__ float gpu_distance(float *v1, float *v2, int len)
{
        float d=0;
        for(int i = 1; i < len+1; i++){
                float tmp = v2[i] - v1[i];
                d += tmp*tmp;
        }
        return d;
}

__global__ void gpu_update_label(int k, int n, int len, float* vec, int* label, float* cen)
{
        int i= blockIdx.x*blockDim.x + threadIdx.x;
        float* vector=vec+i*(len+1);
        float min_d=INF, d;
        int new_label=1021;
        extern __shared__ float center[];
        for(int j = 0; j < k; j++){
                if(threadIdx.x==0)load_center(center, cen+j*(len+1), len);
                __syncthreads();  

                if(i<n){
                        d = gpu_distance(center, vector, len);
                        if(d < min_d){
                                min_d = d;
                                new_label = j;
                        }
                }  
                __syncthreads();
        }
        __syncthreads();
        label[i]=new_label;
}

static void updateCenter(int k, int n, int len, float *vec[],
                  int* label, float* center)
{
        float* tmp_cen;
        memset(center, 0, k*(len+1)*sizeof(float));
        for(int i=0; i<n; i++){
                if(label[i]>=k)printf("label[%d]:%d\n", i, label[i]);
                assert(label[i]<k && label[i]>=0);
                tmp_cen=center+label[i]*(len+1);
                for(int j=0; j<len; j++)
                        tmp_cen[j+1] += vec[i][j];
                tmp_cen[0]++;
        }

        for(int i=0; i<k; i++){
                tmp_cen=center+i*(len+1);
                for(int j=1; j<len+1; j++)
                        tmp_cen[j]/=tmp_cen[0];
        }
}

__global__ void gpu_update_center(int k, int n, int len,  float* vec, float* cen)
{
        int i= blockIdx.x*blockDim.x + threadIdx.x;
        float* vector=vec+i*(len+1);
	float* tmp_cen;
	float* center;
        int label;
        if(i==0){
                for(int j = 0; j < k; j++){
                        tmp_cen=cen+j*(len+1);
			for(int m=0; m<len+1; m++)tmp_cen[m]=0;
                }
        }
        __syncthreads();
	if(i<n){
                label=*((int*)vector);
		center = cen + label*(len+1);
		for(int j=1; j<len+1; j++){
			center[j] += vector[j];
		}
		center[0]++;
	}
        __syncthreads();
        if(i==0){
                for(int j=0; j<k; j++){
                        tmp_cen=cen+j*(len+1);
                        for(int m=1; m<len+1; m++){
                                tmp_cen[m]/=*tmp_cen;
                        }
                }
        }
        __syncthreads();  
}

static void updateLabel(int k, int n, int len, float *vec,int* label,  float* center)
{
        int numThreadsPerBlock=256;
        int numBlocks=n/numThreadsPerBlock;
        gpu_update_label<<<numBlocks, numThreadsPerBlock, len*sizeof(float)>>>(k, n, len, vec, label, center);
        checkCUDAError("updateLabel:");
}

/* static void updateCenter(int k, int n, int len, float *vec, float* center) */
/* { */
/*         int numThreadsPerBlock=256; */
/*         int numBlocks=n/numThreadsPerBlock ; */
/*         gpu_update_center<<<numBlocks, numThreadsPerBlock>>>(k, n, len, vec, center); */
/*         checkCUDAError("updateCenter:"); */
/* } */

__global__ void gpu_copy_label(int n, int len, float* vec, int* lab)
{
        int i= blockIdx.x*blockDim.x + threadIdx.x;
        float* vector=vec+i*(len+1);
        if(i<n)
                lab[i]=*((int*)vector);
}

static void copyLabel(int n, int len, float* vec, int* lab)
{
        int numThreadsPerBlock=256;
        int numBlocks=n/numThreadsPerBlock;
        int* label;
        cudaMalloc((void**)&label, n*sizeof(int));
        gpu_copy_label<<<numBlocks, numThreadsPerBlock>>>(n, len, vec, label);
        checkCUDAError("copyLabel:");
        cudaMemcpy(lab, label, n*sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(label);
}

static void initVec(int n, int len, int* label, float *vec[], float** d_vec)
{
        int i;
        cudaMalloc((void**)d_vec, n*(len+1)*sizeof(float));
        for(i=0; i<n; i++){
                cudaMemcpy((*d_vec)+i*(len+1), label+i, sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy((*d_vec)+i*(len+1)+1, vec[i], len*sizeof(float), cudaMemcpyHostToDevice);
        }
}

static void freeVec(float *d_vec)
{
        cudaFree(d_vec);
}

static void initCenter(int k, int len, float** center, float** d_center)
{
        *center=(float*)malloc(k*(len+1)*sizeof(float));
        if(!*center)panic("no mem!\n");
        cudaMalloc((void**)d_center, k*(len+1)*sizeof(float));
}

static void freeCenter(float* center)
{
        cudaFree(center);
}

static bool isChanged(int n, int *label[])
{
        int i;
        for(i=0; i<n; i++)
                if(label[0][i]!=label[1][i])
                        return true;
        return false;
}

void initLabel(int n, int *lab, int* label[])
{
        label[0]=(int*)malloc(n*sizeof(int));
        label[1]=(int*)malloc(n*sizeof(int));
        if(!label[0] || !label[1])panic("no mem!\n");
        memcpy(label[0], lab, n*sizeof(int));
        memset(label[1], 0, n*sizeof(int));
}

void freeLabel(int* label[])
{
        free(label[0]);
        free(label[1]);
}

void copyLabel(int n,  int* lab, int* label)
{
        memcpy(lab, label, n*sizeof(int));
}

void GPUKmeans::kmeansClustering(int k, int n, int len, float *vec[],
                                 int* lab)
{
        float *center, *d_center;
        int *d_label, *label[2];
        float* d_vec;
        bool changed=true;
        int cur=0, iter;
        initLabel(n, lab, label);
        initCenter(k, len, &center, &d_center);
        initVec(n, len, label[cur], vec, &d_vec);
        cudaMalloc((void**)&d_label, n*sizeof(int));
        for(iter=0; iter<=10 && changed; iter++){
                updateCenter(k, n, len, vec, label[cur], center);
                cudaMemcpy(d_center, center, k*(len+1)*sizeof(float), cudaMemcpyHostToDevice);
		/* vec_print(d_center, 10); */
                updateLabel(k, n, len, d_vec, d_label, d_center);
                cur^=1;
                cudaMemcpy(label[cur], d_label, n*sizeof(int), cudaMemcpyDeviceToHost);
                /* changed=isChanged(n, label); */
        }
        copyLabel(n, lab, label[cur]);
        freeLabel(label);
        freeVec(d_vec);
        freeCenter(center);
        printf("iter: %d\n", iter);
}
