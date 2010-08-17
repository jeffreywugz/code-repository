#!/usr/bin/python

import re
import os

def get_corpus_file_list(corpus):
    file_list=['%s/%s'%(corpus, file_name) for file_name in os.listdir(corpus)]
    file_list.sort()
    return file_list

def file_to_token_list(file_name):
    return re.split('[^a-z]+', open(file_name).read())

