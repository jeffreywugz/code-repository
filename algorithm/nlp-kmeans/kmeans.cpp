#include <math.h>
#include <time.h>
#include "kmeans.h"
#include "kmeans.h"

static int rand_range(int k)
{
        return random()%k;
}

void Kmeans::kmeansInitLabel(int k, int n, int *label)
{
        int i;
        srandom(time(NULL));
        for(i=0; i<n; i++){
                label[i]=rand_range(k);
        }
}

void Kmeans::clusterPrint(int k, int n, int* label)
{
        int* count;
        int i;
        count=(int*)malloc(k*sizeof(int));
        if(!count)panic("no mem!\n");
        memset(count, 0, k*sizeof(int));
        for(i=0; i<n; i++){
                count[label[i]]++;
        }
        for(i=0; i<n; i++){
                printf("%d %d\n", i, label[i]);
        }
        for(i=0; i<k; i++){
                printf("cluster %d: count: %d\n", i, count[i]);
        }
        free(count);
}

void Kmeans::clustering(int max_n_iter, int k, VectorSet* vset, int *label)
{
        kmeansInitLabel(k, vset->n_vector, label);
        kmeansClustering(max_n_iter, k, vset->n_vector, vset->len_vector,
                         vset->vector, label);
        clusterPrint(k, vset->n_vector, label);
}

static inline double* vec_kmul(double *v1, double k, int len)
{
    int i;
    for(i=0; i<len; i++)
	v1[i]*=k;
    return v1;
}

static inline double* vec_add(double *v2, double *v1, int len)
{
    int i;
    for(i=0; i<len; i++)
	v2[i]+=v1[i];
    return v2;
}

static inline double square_distance(double *v1, double *v2, int len)
{
    int i;
    double s;
    for(s=0,i=0; i<len; i++)
	s+=(v1[i]-v2[i])*(v1[i]-v2[i]);
    return s;
}

static inline double sin_distance(double *v1, double *v2, int len)
{
    int i;
    double s;
    for(s=0,i=0; i<len; i++)
	s+=v1[i]*v2[i];
    return 1.0-s;
}

#define distance square_distance
#define DELTA (1e-40)
static int getLabel(int k, int len, double* vec, double* center[])
{
        int i;
        int label;
        double d, min_d=INF;
        for(i=0; i<k; i++){
                d=distance(vec, center[i], len);
                if(d<min_d){
                        label=i;
                        min_d=d;
                }
        }
        return label;
}

static void updateLabel(int k, int n, int len, double *vec[],
                 int* label, double* center[])
{
        int i;
        for(i=0; i<n; i++){
                label[i]=getLabel(k, len, vec[i], center);
        }
}

static void updateCenter(int k, int n, int len, double *vec[],
                  int* label, double* center[])
{
        int i;
        int* count;
        count=(int*)malloc(k*sizeof(int));
        if(!count)panic("no mem!\n");
        memset(count, 0, k*sizeof(int));
        for(i=0; i<k; i++)
                memset(center[i], 0, len*sizeof(double));
        
        for(i=0; i<n; i++){
                vec_add(center[label[i]], vec[i], len);
                count[label[i]]++;
        }

        for(i=0; i<k; i++){
                vec_kmul(center[i], 1.0/count[i], len);
        }
        free(count);
}


static bool isChanged(int k, int len, double** center[])
{
        int i;
        double s=0;
        for(i=0; i<k; i++){
                s += distance(center[0][i], center[1][i], len);
        }
        printf("%f:\n", s);
        return s>DELTA;
}

static void initCenter(int k, int len, double** center[])
{
        int i;
        center[0]=(double**)malloc(k*sizeof(double*));
        center[1]=(double**)malloc(k*sizeof(double*));
        if(!center[0] || !center[1])panic("no mem!\n");
        for(i=0; i<k; i++){
                center[0][i]=(double*)malloc(len*sizeof(double));
                center[1][i]=(double*)malloc(len*sizeof(double));
                memset(center[0][i], 0, len*sizeof(double));
                memset(center[1][i], 0, len*sizeof(double));
        }
}

static void freeCenter(int k, double** center[])
{
        int i;
        for(i=0; i<k; i++){
                free(center[0][i]);
                free(center[1][i]);
        }
        free(center[0]);
        free(center[1]);
}

void Kmeans::kmeansClustering(int max_n_iter, int k, int n, int len, double *vec[],
                                 int* label)
{
        double **center[2];
        bool changed=true;
        int cur, iter;
        initCenter(k, len, center);
        for(iter=0, cur=0; iter<=max_n_iter && changed; iter++, cur^=1){
                updateCenter(k, n, len, vec, label, center[cur]);
                updateLabel(k, n, len, vec, label, center[cur]);
                changed=isChanged(k, len, center);
        }
        printf("n_iter: %d\n", iter);
        freeCenter(k, center);
}

