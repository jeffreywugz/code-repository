#!/bin/env python2
'''
./blktrace-sort.py blktrace.log
'''

from itertools import groupby
import sys
import re

def parse_blk(raw):
    pat = r'^\s*n,n n n ([.0-9]+) n ([A-Z]+) [A-Z]+ '.replace('n', r'\d+').replace(' ', r'\s+') + r'(\d+ [+] \d+)'
    return re.findall(pat, raw, re.M)

def extract(iter):
    result = []
    ts0, op0, pos0 = iter.next()
    min_ts, max_ts = ts0, ts0
    for ts, op, pos in iter:
        latency, ops = float(ts) - float(ts0), '%s-%s'%(op0, op)
        result.append((latency, ops, ts0))
        ts0, op0, pos0 = ts, op, pos
        max_ts = ts0
    return float(max_ts)-float(min_ts), result

def sort_blk(raw):
    trace = parse_blk(raw)
    trace.sort(key=lambda (ts, op, pos): (pos, float(ts)))
    grouped_trace = [(pos, extract(subiter)) for pos, subiter in groupby(trace, lambda (ts, op, pos): pos)]
    grouped_trace.sort(key=lambda (pos, (latency, evlist)): latency, reverse=True)
    return grouped_trace
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print __doc__
        sys.exit(1)
    all_ev_list = []
    for (pos, (max_latency, evlist)) in sort_blk(file(sys.argv[1]).read()):
        all_ev_list.extend([(pos.replace(' ', ''), eid, latency, ts) for (latency,eid, ts) in evlist])
    all_ev_list.sort(key=lambda (pos,eid,latency, ts): latency, reverse=True)
    print 'pos:str ev:str duration:float ts:float'
    for ev in all_ev_list:
        print '\t'.join(map(str, ev))
