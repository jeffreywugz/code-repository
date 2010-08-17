#include <math.h>
#include "kmeans.h"

static inline float* vec_kmul(float *v1, float k, int len)
{
    int i;
    for(i=0; i<len; i++)
	v1[i]*=k;
    return v1;
}

static inline float* vec_add(float *v2, float *v1, int len)
{
    int i;
    for(i=0; i<len; i++)
	v2[i]+=v1[i];
    return v2;
}

static inline float distance(float *v1, float *v2, int len)
{
    int i;
    float s;
    for(s=0,i=0; i<len; i++)
	s+=(v1[i]-v2[i])*(v1[i]-v2[i]);
    return s;
}

static void vec_print(float *vec, int len)
{
    for(int i=0; i<10; i++)
            printf("%f ", vec[i]);
    printf("\n");
}

#define DELTA (1e0)
static int getLabel(int k, int len, float* vec, float* center[])
{
        int i;
        int label;
        float d, min_d=INF;
        for(i=0; i<k; i++){
                d=distance(vec, center[i], len);
                if(d<min_d){
                        label=i;
                        min_d=d;
                }
        }
        return label;
}

static void updateLabel(int k, int n, int len, float *vec[],
                 int* label, float* center[])
{
        int i;
        for(i=0; i<n; i++){
                label[i]=getLabel(k, len, vec[i], center);
        }
}

static void updateCenter(int k, int n, int len, float *vec[],
                  int* label, float* center[])
{
        int i;
        int* count;
        count=(int*)malloc(k*sizeof(int));
        if(!count)panic("no mem!\n");
        memset(count, 0, k*sizeof(int));
        for(i=0; i<k; i++)
                memset(center[i], 0, len*sizeof(float));
        
        for(i=0; i<n; i++){
                vec_add(center[label[i]], vec[i], len);
                count[label[i]]++;
        }

        for(i=0; i<k; i++){
                vec_kmul(center[i], 1.0/count[i], len);
        }
        free(count);
}


static bool isChanged(int k, int len, float** center[])
{
        int i;
        float s=0;
        for(i=0; i<k; i++){
                s += distance(center[0][i], center[1][i], len);
        }
        return s>DELTA;
}

static void initCenter(int k, int len, float** center[])
{
        int i;
        center[0]=(float**)malloc(k*sizeof(float*));
        center[1]=(float**)malloc(k*sizeof(float*));
        if(!center[0] || !center[1])panic("no mem!\n");
        for(i=0; i<k; i++){
                center[0][i]=(float*)malloc(len*sizeof(float));
                center[1][i]=(float*)malloc(len*sizeof(float));
                memset(center[0][i], 0, len*sizeof(float));
                memset(center[1][i], 0, len*sizeof(float));
        }
}

static void freeCenter(int k, float** center[])
{
        int i;
        for(i=0; i<k; i++){
                free(center[0][i]);
                free(center[1][i]);
        }
        free(center[0]);
        free(center[1]);
}

void CPUKmeans::kmeansClustering(int k, int n, int len, float *vec[],
                                 int* label)
{
        float **center[2];
        bool changed=true;
        int cur, iter;
        initCenter(k, len, center);
        for(iter=0, cur=0; iter<=10 && changed; iter++, cur^=1){
                updateCenter(k, n, len, vec, label, center[cur]);
		vec_print(center[cur][0], 10);
                updateLabel(k, n, len, vec, label, center[cur]);
                changed=isChanged(k, len, center);
        }
        freeCenter(k, center);
}

