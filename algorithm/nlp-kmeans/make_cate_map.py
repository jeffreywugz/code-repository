#!/usr/bin/python

def write_cate_map(cate_map, cate_id_list):
    file=open(cate_map, 'w')
    lines=['%s %s\n'%(item[0], ' '.join(item[1])) for item in cate_id_list]
    file.writelines(lines)

def get_file_map(file_map):
    lines=open(file_map).readlines()
    lines=[item.split() for item in lines]
    lines=[(item[0].split('/')[-1], item[1]) for item in lines]
    return dict(lines)
    
def make_cate_map(categories, file_map, cate_map):
    cate_list=[item.split() for item in open(categories).readlines()]
    file_map=get_file_map(file_map)
    cate_id_list=[]
    for item in cate_list:
        cate=item[0]
        files=item[1:]
        fileids=[file_map[item] for item in files]
        cate_id_list.append((cate, fileids))

    write_cate_map(cate_map, cate_id_list)

