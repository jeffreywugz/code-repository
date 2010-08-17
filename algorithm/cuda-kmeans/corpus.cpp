#include "kmeans.h"

Corpus::Corpus(const char* file)
{
        strcpy(this->file, file);
}

Corpus::~Corpus()
{
}

#define align(x,n) (x+n-1)&~(n-1)
bool Corpus::read(VectorSet* vset)
{
        FILE* fp;
        int n_doc, n_word, n_line;
        fp=fopen(file, "r");
        if(!fp)panic("can't read corpus: %s\n", file);
        fscanf(fp, "%d%d%d", &n_doc, &n_word, &n_line);
	n_word=align(n_word, 256);
        if(!vset->init(n_doc, n_word))return false;
        fillVector(n_line, fp, vset);
        return true;
}

bool Corpus::fillVector(int n_line, FILE* fp, VectorSet* vset)
{
        int doc_id, word_id, word_count;
        int i;
        for(i=0; i<n_line; i++){
                assert(fscanf(fp, "%d%d%d", &doc_id, &word_id, &word_count)==3);
                doc_id %= vset->n_vector;
                word_id %= vset->len_vector;
                assert(doc_id<vset->n_vector && doc_id>=0);
                assert(word_id<vset->len_vector && word_id>=0);
                vset->vector[doc_id][word_id]=word_count;
        }
        return true;
}
        
