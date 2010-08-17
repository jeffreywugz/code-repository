#include "kmeans.h"

Corpus::Corpus(const char* file, const char* config)
{
        strcpy(this->file, file);
        strcpy(this->config, config);
}

Corpus::~Corpus()
{
}

#define align(x,n) (x+n-1)&~(n-1)
bool Corpus::read(VectorSet* vset)
{
        FILE *file_fp, *config_fp;
        int n_doc, n_word, n_line;
        config_fp=fopen(config, "r");
        if(!config_fp)panic("can't read corpus: %s\n", config);
        fscanf(config_fp, "%d%d%d", &n_doc, &n_word, &n_line);
        fclose(config_fp);

        printf("n_doc: %d\n", n_doc);
        printf("n_word: %d\n", n_word);
        printf("n_line: %d\n", n_line);
        
        file_fp=fopen(file, "r");
        if(!vset->init(n_doc, n_word))return false;
        fillVector(n_line, file_fp, vset);
        fclose(file_fp);
        return true;
}

bool Corpus::fillVector(int n_line, FILE* fp, VectorSet* vset)
{
        int doc_id, word_id, word_count;
        int i;
        for(i=0; i<n_line; i++){
                assert(fscanf(fp, "%d%d%d", &doc_id, &word_id, &word_count)==3);
                assert(doc_id<vset->n_vector && doc_id>=0);
                assert(word_id<vset->len_vector && word_id>=0);
                vset->vector[doc_id][word_id]=word_count;
        }
        return true;
}
        
