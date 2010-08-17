#include <stdio.h>
#include <sys/time.h>
#include "kmeans.h"

void test_kmeans(char *msg, int max_n_iter, int k,
                 Kmeans* kmeans, VectorSet* vset)
{
        int* label;
        struct timeval start, end;
        double timeuse;
        printf("######## %s #########\n", msg);
        label=(int*)malloc(vset->n_vector*sizeof(int));
        if(!label)panic("no mem!\n");
        gettimeofday(&start, NULL);
        kmeans->clustering(max_n_iter, k, vset, label);
        gettimeofday(&end, NULL);
        timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) +
                end.tv_usec - start.tv_usec;
        timeuse /= 1000000;
        printf("time usage:%f ms\n", timeuse);
        free(label);
}


int main(int argc, char* argv[])
{
        if(argc!=2)panic("usage: %s config_file", argv[0]);
        Config config(argv[1]);
        config.print();
        Corpus corpus(config.corpus_file, config.corpus_config);
        VectorSet vset;
        corpus.read(&vset);
        vset.tf_idf();
        // vset.normalize();
        Kmeans kmeans;
        test_kmeans("test kmeans:", config.max_n_iter, config.n_cluster,
                    &kmeans, &vset);
        return 0;
}
