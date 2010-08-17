#!/usr/bin/python

import config
import exceptions
import os.path
from corpus_reader import *

def add_list_to_dict(words_dict, words_list):
    for word in words_list:
        try:
            words_dict[word] += 1
        except exceptions.Exception,e:
            words_dict[word] = 1
            
def add_file_to_dict(words_dict, file_name):
    add_list_to_dict(words_dict, file_to_token_list(file_name))

def corpus_to_dict(corpus):
    words_dict={}
    file_list=get_corpus_file_list(corpus)
    for file_name in file_list:
        add_file_to_dict(words_dict, file_name)
    return words_dict

def filter_words(words_dict, min_count, max_count):
    words_list=words_dict.items()
    words_list=[x for x in words_list if x[1]>10 and x[1]<1000]
    words_list.sort(key=lambda x:x[1])
    return words_list
    
def corpus_to_words_list(raw_corpus, words_list_file):
    if os.path.exists(words_list_file):
        return
    words_dict=corpus_to_dict(config.raw_corpus)
    words_list=filter_words(words_dict, 10, 1000)
    lines=['%s %s\n'%(item[0], item[1]) for item in words_list]
    open(words_list_file, 'w').writelines(lines)
    
if __name__ == '__main__':
    corpus_to_words_list(config.raw_corpus, config.words_list)
