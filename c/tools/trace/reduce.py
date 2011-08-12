#!/usr/bin/env python2
'''
Usages:
./reduce exefile plain <trace.out >trace.plain
./reduce exefile dot <trace.out >trace.dot
'''

import sys
import re
from subprocess import Popen, PIPE, STDOUT

def info(msg):
    sys.stderr.write(msg)
    
def err(msg, rc=1):
    sys.stderr.write(msg)
    sys.exit(rc)

def getitem(seq, i, default=None):
    return i in range(len(seq)) and seq[i] or default

def popen(cmd, input=None):
    return Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT).communicate(input=input)[0]

def addr2line(exe, addrs):
    return popen('addr2line -e %s -f -C'%(exe), '\n'.join(addrs))

def transpose():
    pass

def plain(exe):
    trace = [i.split() for i in sys.stdin.readlines()]
    fn_trace = re.findall('^(.+)\n(.+)$', addr2line(exe, [addr for enter_exit, addr in trace]), re.M)

def dot(exe):
    pass

if __name__ == '__main__':
    len(sys.argv) in range(2,4) or err(__doc__)
    exefile, term = sys.argv[1], getitem(sys.argv, 2, "dot")
    info("%s(exefile='%s')\n"%(term, exefile))
    term_err =  lambda *arg,**kw: err('no term defined for %s\n'%term)
    print (globals().get(term) or term_err)(exefile)

