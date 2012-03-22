#!/usr/bin/env python2
'''
./plot.py input.data output.img
'''
import sys
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def parse_cmd_args(args):
    list_args = [i for i in args if not re.match('\w+=', i)]
    kw_args = dict(i.split('=', 1) for i in args if re.match('\w+=', i))
    return list_args, kw_args

def read(path):
    with open(path, 'r') as f:
        return f.read()
    
def plt_dump(path):
    if path:
        plt.savefig(path)
    else:
        plt.show()

def parse_float(pat, data):
    return [map(float, items) for items in re.findall(pat, data)]

def diff_y(x, y):
    return x[1:], np.diff(y)

def parse_float_from_file(pat, file=None):
    return parse_float(pat, file and read(file) or sys.stdin.read())

def diff_plot_file(pat, img, file=None, style='.', xlabel='x', ylabel='y'):
    x, y = np.transpose(parse_float_from_file(pat, file))
    x, y = diff_y(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y, style)
    plt_dump(img)
    return file

def help():
    print __doc__
    sys.exit(-1)

if __name__ == '__main__':
    len(sys.argv) <= 1 and help()
    args, kw = parse_cmd_args(sys.argv[2:])
    print globals().get(sys.argv[1], help)(*args, **kw)
