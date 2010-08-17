#include "kmeans.h"

Config::Config(const char* config_file)
{
        char corpus_file[256], corpus_config[256], n_cluster[256], max_n_iter[256];
        ConfigFile config(config_file);
        if(!config.getVal("corpus_file", corpus_file))
                panic("config error:%s!\n", "corpus_file");
        if(!config.getVal("corpus_config", corpus_config))
                panic("config error:%s!\n", "corpus_config");
        if(!config.getVal("n_cluster", n_cluster))
                panic("config error:%s!\n", "n_cluster");
        if(!config.getVal("max_n_iter", max_n_iter))
                panic("config error:%s!\n", "max_n_iter");
        strcpy(this->corpus_file, corpus_file);
        strcpy(this->corpus_config, corpus_config);
        this->n_cluster=atoi(n_cluster);
        this->max_n_iter=atoi(max_n_iter);
}

void Config::print()
{
        printf("corpus_file: %s\n", corpus_file);
        printf("corpus_config: %s\n", corpus_config); 
        printf("n_cluster: %d\n", n_cluster);
        printf("max_n_iter: %d\n", max_n_iter);
}
