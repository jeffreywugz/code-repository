#include "kmeans.h"

bool VectorSet::init(int n_vector, int len_vector)
{
        int i;
        float *p;
        this->n_vector=n_vector;
        this->len_vector=len_vector;
        vector=(float**)malloc(n_vector*sizeof(float*));
        if(!vector)panic("no mem!\n");
        p=(float*)malloc(n_vector*len_vector*sizeof(float));
        memset(p, 0, n_vector*len_vector*sizeof(float));
        if(!p)panic("no mem!\n");
        for(i=0; i<n_vector; i++){
                vector[i] = p + i*len_vector;
        }
        return true;
}
