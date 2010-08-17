#include "kmeans.h"

Config::Config(const char* config_file)
{
        char corpus_file[256], n_cluster[256];
        ConfigFile config(config_file);
        if(!config.getVal("corpus_file", corpus_file))panic("config error!\n");
        if(!config.getVal("n_cluster", n_cluster))panic("config error!\n");
        strcpy(this->corpus_file, corpus_file);
        this->n_cluster=atoi(n_cluster);
}

void Config::print()
{
        printf("corpus_file: %s\n", corpus_file);
        printf("n_cluster: %d\n", n_cluster);
}
