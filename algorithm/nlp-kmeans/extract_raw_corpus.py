#!/usr/bin/python
import config
import os.path

def extract_raw_corpus(nltk_data, raw_corpus, categories):
    if os.path.exists(raw_corpus):
        return
    os.mkdir(raw_corpus)
    import nltk
    nltk.data.path.append(nltk_data)
    corpus=nltk.corpus.brown
    for file_name in corpus.fileids():
        raw_file=open('%s/%s'%(config.raw_corpus, file_name), 'w')
        raw_file.write(' '.join(corpus.words(fileids=file_name)))
    
    cate_file=open(categories, 'w')
    for cate in corpus.categories():
        cate_file.write('%s '%cate)
        cate_file.write(' '.join(corpus.fileids(categories=cate)))
        cate_file.write('\n')
        
        

if __name__ == '__main__':
    extract_raw_corpus(config.nltk_data, config.raw_corpus, config.categories)
        
