#include <stdio.h>
#include <sys/time.h>
#include "kmeans.h"

// function to test a class which implement k-means clustering algorithm
void test_kmeans(char *msg, int k, Kmeans* kmeans, VectorSet* vset)
{
        int* label;
        struct timeval start, end;
        double timeuse;
        printf("######## %s #########\n", msg);
        label=(int*)malloc(vset->n_vector*sizeof(int));
        if(!label)panic("no mem!\n");
        gettimeofday(&start, NULL);
        kmeans->clustering(k, vset, label);
        gettimeofday(&end, NULL);
        timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
        timeuse /= 1000000;
        printf("time usage:%f ms\n", timeuse);
        free(label);
}


int main(int argc, char* argv[])
{
        // check run arguments
        if(argc!=2)panic("usage: %s config_file", argv[0]);
        // read config file
        Config config(argv[1]);
        config.print();
        // init corpus
        Corpus corpus(config.corpus_file);
        VectorSet vset;
        // init vector set from corpus
        corpus.read(&vset);
        CPUKmeans cpukmeans;    // class inplement k-means on cpu
        GPUKmeans gpukmeans;    // class inplement k-means on gpu
        test_kmeans("cpu kmeans:", config.n_cluster, &cpukmeans, &vset); // test k-means on cpu
        test_kmeans("gpu kmeans:", config.n_cluster, &gpukmeans, &vset); // test k-means on gpu
        return 0;
}
