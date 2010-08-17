#!/usr/bin/python

import config
from corpus_reader import *
import exceptions
import os.path


def write_words_bag_config(config_file, n_doc, n_word, n_line):
    open(config_file, 'w').write('%d %d %d\n'%(n_doc, n_word, n_line))

def get_file_map(corpus):
    file_list=get_corpus_file_list(corpus)
    return dict(map(None, file_list, range(len(file_list))))

def write_file_map(file_map_file, file_map):
    if os.path.exists(file_map_file):
        return
    file=open(file_map_file, 'w')
    file_list=file_map.items()
    file_list.sort(key=lambda x: x[1])
    lines=['%s %s\n'%item for item in file_list]
    file.writelines(lines)

def get_word_map(words_list_file):
    words_list=get_words_list(words_list_file)
    return dict(map(None, words_list, range(len(words_list))))

def write_word_map(word_map_file, word_map):
    if os.path.exists(word_map_file):
        return
    file=open(word_map_file, 'w')
    word_list=word_map.items()
    word_list.sort(key=lambda x: x[1])
    lines=['%s %s\n'%item for item in word_list]
    file.writelines(lines)
    
def get_words_list(words_list):
    lines=open(words_list).readlines()
    items=[line.split(' ') for line in lines]
    return [item[0] for item in items]

def words_list_to_words_dict(words_list):
    words_dict={}
    for word in words_list:
        try:
            words_dict[word] += 1
        except exceptions.Exception,e:
            words_dict[word]=1
    return words_dict

def add_to_global_words_list(global_words_list, file_id, word_map, words_dict):
    word_item=map(lambda (word, count): (file_id, word_map[word], count), words_dict.items())
    word_item.sort(key=lambda x: x[1])
    global_words_list.extend(word_item)

def write_words_bag(words_bag, global_words_list):
    words_bag_file=open(words_bag,'w')
    lines=['%d %d %d\n'%(item[0], item[1], item[2]) for item in global_words_list]
    words_bag_file.writelines(lines)
    
def make_words_bag(raw_corpus, words_list, words_bag, word_map_file, file_map_file, config_file):
    if os.path.exists(words_bag):
        return
    file_map=get_file_map(raw_corpus)
    word_map=get_word_map(words_list)
    global_words_list=[]
    for file_name in get_corpus_file_list(raw_corpus):
        _words_list=file_to_token_list(file_name)
        _words_list=[word for word in _words_list if word_map.has_key(word)]
        words_dict=words_list_to_words_dict(_words_list)
        add_to_global_words_list(global_words_list, file_map[file_name], word_map, words_dict)
    write_words_bag(words_bag, global_words_list)
    write_word_map(word_map_file, word_map)
    write_file_map(file_map_file, file_map)
    write_words_bag_config(config_file, len(file_map), len(word_map), len(global_words_list))

