#!/usr/bin/python

import config
import extract_raw_corpus
import corpus_to_words_list
import make_words_bag
import make_cate_map

def make_all():
    extract_raw_corpus.extract_raw_corpus(config.nltk_data, config.raw_corpus, config.categories)
    corpus_to_words_list.corpus_to_words_list(config.raw_corpus, config.words_list)
    make_words_bag.make_words_bag(config.raw_corpus, config.words_list,
                                  config.words_bag, config.word_map, config.file_map,config.words_bag_config)
    make_cate_map.make_cate_map(config.categories, config.file_map, config.cate_map)

if __name__ == '__main__':
    make_all()    
