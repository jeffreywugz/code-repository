#include "kmeans.h"
#include <math.h>

bool VectorSet::init(int n_vector, int len_vector)
{
        int i;
        double *p;
        this->n_vector=n_vector;
        this->len_vector=len_vector;
        vector=(double**)malloc(n_vector*sizeof(double*));
        if(!vector)panic("no mem!\n");
        p=(double*)malloc(n_vector*len_vector*sizeof(double));
        memset(p, 0, n_vector*len_vector*sizeof(double));
        if(!p)panic("no mem!\n");
        for(i=0; i<n_vector; i++){
                vector[i] = p + i*len_vector;
        }
        return true;
}

void VectorSet::tf_idf()
{
        double *idf;
        int i, j;
        int d_count;
        idf=(double*)malloc(len_vector*sizeof(double));
        if(!idf)panic("no mem!\n");
        memset(idf, 0, len_vector*sizeof(double));
        for(i=0; i<len_vector; i++){
                d_count=0;
                for(j=0; j<n_vector; j++){
                        if(vector[j][i] > 0.1)
                                d_count++;
                }
                idf[i]=d_count;
        }

        for(j=0; j<len_vector; j++)
                idf[j] = log(n_vector/idf[j]);
        for(i=0; i<n_vector; i++){
                for(j=0; j<len_vector; j++){
                        vector[i][j] *= idf[j];
                }
        }
}

void VectorSet::normalize()
{
        int i, j;
        double norm;
        for(i=0; i<n_vector; i++){
                norm=0.0;
                for(j=0; j<len_vector; j++)
                        norm += vector[i][j] * vector[i][j];
                norm=sqrt(norm);
                for(j=0; j<len_vector; j++)
                        vector[i][j] /= norm;
        }
}
