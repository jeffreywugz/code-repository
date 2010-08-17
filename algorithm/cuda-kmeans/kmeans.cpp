#include "kmeans.h"

static int rand_range(int k)
{
        return random()%k;
}

void Kmeans::kmeansInitLabel(int k, int n, int *label)
{
        int i;
        srandom(314);
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
        for(i=0; i<k; i++){
                printf("cluster %d: count: %d\n", i, count[i]);
        }
        free(count);
}

void Kmeans::clustering(int k, VectorSet* vset, int *label)
{
        kmeansInitLabel(k, vset->n_vector, label);
        kmeansClustering(k, vset->n_vector, vset->len_vector,
                         vset->vector, label);
        clusterPrint(k, vset->n_vector, label);
}

