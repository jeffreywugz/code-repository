#!/usr/bin/python

import config
import sys
import operator

def get_cate_map(cate_map_file):
    lines=open(cate_map_file).readlines()
    lines=[item.split() for item in lines]
    return dict([(item[0], item[1:]) for item in lines])

def get_result_map(result_file):
    lines=open(result_file).readlines()
    return dict([item.split() for item in lines])

def get_max_count(files, result_map):
    cate_list=[cate for (file,cate) in result_map.items() if file in files]
    max_count=max([operator.countOf(cate_list, item) for item in set(cate_list)])
    return max_count
        
        
def analyze(cate_map_file, result_file):
    cate_map=get_cate_map(cate_map_file)
    result_map=get_result_map(result_file)
    cate_precision=[]
    for cate,files in cate_map.items():
        max_count=get_max_count(files, result_map)
        cate_precision.append(1.0*max_count/len(files))
        # print '%s\t\t%d\t%d\t%f'%(cate, len(files), max_count, 1.0*max_count/len(files))
    return cate_precision
                
if __name__ == '__main__':
    precision_list=analyze(config.cate_map, sys.argv[1])
    step_list=[x*0.5 for x in range(15)]
    for step,precision in map(None, step_list, precision_list):
        print '(%s,%.2f) '%(step, precision),
    
        
        
