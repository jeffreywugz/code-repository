#ifndef _KMEANS_H_
#define _KMEANS_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "debug.h"

#define INF (1e20)

class ConfigFile
{
public:
        char file[256];
        ConfigFile(const char* config_file);
        ~ConfigFile();
        bool getVal(const char* key, char* val);
        void print();
private:
        char buf[1024];
        char* dict[128][2];
        int n_item;
        void fillDict();
        int getLines(char* lines[]);
        bool parseLine(char* line, char* &key, char* &val);
};

class Config
{
public:
        char corpus_file[256];
        char corpus_config[256];
        int n_cluster;
        int max_n_iter;
        Config(const char* file);
        void print();
};

class VectorSet
{
public:
        int n_vector;
        int len_vector;
        double** vector;
        bool init(int n_vector, int len_vector);
        void tf_idf();
        void normalize();
};

class Corpus
{
public:
        Corpus(const char* file, const char* config);
        ~Corpus();
        char file[256];
        char config[256];
        bool read(VectorSet* vset);
private:
        bool fillVector(int n_line, FILE* fp, VectorSet* vset);

};

class Kmeans
{
public:
        void clustering(int max_n_iter, int k, VectorSet* vset, int* label);
private:
        void kmeansInitLabel(int k, int n, int *label);
        void clusterPrint(int k, int n, int* label);
        void kmeansClustering(int max_n_iter, int k, int n, int len, double* vec[],
                                      int *label);
};

#endif /* _KMEANS_H_ */
