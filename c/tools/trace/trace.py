#!/usr/bin/env python2
'''
Usages:
export CC="gcc -finstrument-functions -g" && $(CC) a.c -o a.out
TRACE=trace.png ./trace.py ./a.out
TRACE=trace.plain ./trace.py ./a.out
'''
trace_so_source = r"""
#include <stdio.h>
#include <stdlib.h>

static FILE *g_trace_output;
void trace_init()__attribute__((no_instrument_function, constructor));
void trace_destroy() __attribute__((no_instrument_function, destructor));

#define trace_output(...) if(g_trace_output)fprintf(g_trace_output, __VA_ARGS__)
#define error(...)  { fprintf(stderr, __VA_ARGS__); perror(NULL); }
void trace_init()
{
        char exe[1024];
        int count;
        if(!(g_trace_output = fopen(getenv("TRACE"), "w"))){
                error("fopen('%s')=>NULL: ", getenv("TRACE"));
                return;
        }
        if((count = readlink("/proc/self/exe", exe, sizeof(exe))) < 0)
                return;
        exe[count] = 0;
        trace_output("%s\n", exe);
}

void trace_destroy()
{
        if(g_trace_output)
                fclose(g_trace_output);
}

void __cyg_profile_func_enter( void *this, void *callsite)
{
        trace_output("E %p\n", this);
}

void __cyg_profile_func_exit( void *this, void *callsite)
{
        trace_output("X %p\n", this);
}
"""

import sys
import re
import os, os.path
import math
from subprocess import call, Popen, PIPE, STDOUT
from collections import Counter

def info(msg):
    sys.stderr.write(msg)
    
def err(msg, rc=1):
    sys.stderr.write(msg)
    sys.exit(rc)

def read(path):
    with open(path) as f:
        return f.read()
    
def write(path, content):
    with open(path, 'w') as f:
        f.write(content)
        
html_escape_table = {
    "&": "&amp;",
    '"': "&quot;",
    "'": "&apos;",
    ">": "&gt;",
    "<": "&lt;",
    }

def html_escape(text):
    """Produce entities within text."""
    return "".join(html_escape_table.get(c,c) for c in text)

def getitem(seq, i, default=None):
    return i in range(len(seq)) and seq[i] or default

def accmulate(func, seq, initial):
    def handle(x, i):
        x.append(func(x[-1], i))
        return x
    return reduce(handle, seq, [initial])

def popen(cmd, input=None):
    return Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT).communicate(input=input)[0]

def addr2line(exe, addrs):
    return popen('addr2line -f -C -e %s'%(exe), '\n'.join(addrs))

def last_n_part_of_path(path, n):
    return '/'.join(path.split('/')[-n:])

def common_prefix(ls, maxdepth=2):
    prefix = re.sub('/[^/]*$', '/', os.path.commonprefix(ls))
    return [last_n_part_of_path(i[len(prefix):], maxdepth) for i in ls]

def addr_translate(trace, exe, maxdepth=2):
    enter_exit, addrs = zip(*trace)
    fn_trace = re.findall('^(.+)\n(.+)$', addr2line(exe, addrs), re.M)
    funcs, lineno = zip(*fn_trace)
    return zip(enter_exit, funcs, common_prefix(lineno, maxdepth))

def call_chains(trace):
    def call_link((_enter_exit, parent), (enter_exit, func, lineno)):
        return enter_exit, enter_exit == 'E' and (parent, (func, lineno)) or parent[0]
    return [chain for enter_exit, chain in accmulate(call_link, trace, (None, (None,None))) if enter_exit == 'E']

def plain(trace, exe, maxdepth=2):
    return Counter((parent, func) for ((_parent,parent), func) in call_chains(addr_translate(trace, exe, maxdepth)))

def trace2plain(trace, exe, maxdepth=2):
    def func_repr(func):
        return func and '@'.join(func)
    return '\n'.join('%d %s => %s'%(count, func_repr(caller), func_repr(callee)) for (caller, callee), count in plain(trace, exe, maxdepth).items())

def trace2dot(trace, exe, maxdepth=2):
    def func_repr(func):
        return func and '@'.join(func)
    def call_link2dot(caller, callee, count):
        edge = '''"%(caller)s"->"%(callee)s"[label="%(count)d", fontsize=%(fontsize)d, penwidth=%(linewidth)d];
 "%(callee)s"[shape=plaintext, style=filled, bgcolor=lightgray,
   label=< <font face="monospace" color="black" point-size="%(fontsize)d">%(escaped_func)s</font>
   <br/><font color="navy" point-size="%(smallfontsize)d">%(escaped_lineno)s</font> >];'''
        func, lineno = callee
        return edge %dict(caller=func_repr(caller), callee=func_repr(callee), count=count, escaped_func=html_escape(func), escaped_lineno=html_escape(lineno), 
                          fontsize=10*math.log(3*count), smallfontsize=8*math.log(3*count), linewidth=math.log(3*count))
    digraph = '''
digraph G {
node[shape=box, style=filled];
rankdir=LR;
%s
};'''
    return digraph% '\n'.join(call_link2dot(caller, callee, count) for (caller, callee),count in plain(trace, exe, maxdepth).items())

def create_trace_so(dir):
    write('%s/trace.c'%(dir), trace_so_source)
    return call('gcc -fPIC -rdynamic --shared %(dir)s/trace.c -o %(dir)s/trace.so'%dict(dir=dir), shell=True) == 0

def traced_call(trace_file, trace_so, args):
    os.putenv('LD_PRELOAD', '%s/trace.so'%(cwd))
    os.putenv('TRACE', trace_file)
    call(args)
    os.putenv('LD_PRELOAD', '')

def trace_term_plain(trace_content, exefile, outfile):
    write(outfile, trace2plain(trace_content, exefile))

def trace_term_dot(trace_content, exefile, outfile):
    write(outfile, trace2dot(trace_content, exefile))
    
def trace_term_png(trace_content, exefile, outfile):
    print popen('dot -Tpng -o %s'% outfile, input=trace2dot(trace_content, exefile))
    
if __name__ == '__main__':
    len(sys.argv) > 1 or err(__doc__)
    trace_output = os.getenv('TRACE') or os.path.basename(sys.argv[1]) + '.png'
    trace_basename, trace_ext = os.path.splitext(trace_output)
    trace_term = trace_ext and trace_ext[1:]
    cwd = os.path.dirname(os.path.realpath(__file__))
    os.path.exists('%s/trace.so'%(cwd)) or create_trace_so(cwd) or err('create "%s/trace.so" failed.\n'%(cwd))
    traced_call('%s.trace'% trace_basename, '%s/trace.so'% cwd, sys.argv[1:])
    with open('%s.trace'%(trace_basename)) as f:
        exefile, trace_content = f.readline().strip(), [i.split() for i in f.readlines()]
    _term = globals().get('trace_term_%s'%(trace_term)) or (lambda trace_content, exefile, outfile: err('no term defined for %s'% trace_term))
    _term(trace_content, exefile, trace_output)
