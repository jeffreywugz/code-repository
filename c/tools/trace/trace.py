#!/usr/bin/env python2
'''
Usages:
export CC="gcc -finstrument-functions -g" && $(CC) a.c -o a.out
TRACE=trace.plain ./trace.py ./a.out
TRACE=trace.svg EXEFILE=./a.out ./trace.py
'''
trace_so_source = r"""
#include <sys/mman.h>
#include <fcntl.h>
#include <stdlib.h>

#define MAX_N_TRACE_ITEM (1<<20)

void panic(const char* msg)
{
        perror(msg);
        exit(-1);
}

void** g_trace_items;
void** g_trace_items_current;
int g_trace_fd;
__attribute__((no_instrument_function, constructor)) void trace_init()
{
        long buf_len = MAX_N_TRACE_ITEM * sizeof(void*);
        if((g_trace_fd = open(getenv("TRACE"), O_RDWR|O_CREAT, S_IRWXU)) < 0) panic("open");
        if(ftruncate(g_trace_fd, buf_len) < 0) panic("ftruncate");
        if(!(g_trace_items = mmap(0, buf_len, PROT_READ|PROT_WRITE, MAP_SHARED, g_trace_fd, 0)))
                panic("mmap");
        g_trace_items_current = g_trace_items;
}

__attribute__((no_instrument_function, destructor)) void trace_destroy()
{
        long buf_len = MAX_N_TRACE_ITEM * sizeof(void*);
        munmap(g_trace_items, buf_len);
        ftruncate(g_trace_fd, sizeof(void*) * (g_trace_items_current - g_trace_items));
        close(g_trace_fd);
}

void __cyg_profile_func_enter( void *this, void *callsite)
{
        (g_trace_items_current < g_trace_items + MAX_N_TRACE_ITEM -1) && (*g_trace_items_current++ = this);
}

void __cyg_profile_func_exit( void *this, void *callsite)
{
        (g_trace_items_current < g_trace_items + MAX_N_TRACE_ITEM -1) && (*g_trace_items_current++ = ~0L);
}
"""

import sys
import re
import os, os.path
import signal
import math
from subprocess import call, Popen, PIPE, STDOUT
from collections import Counter
import array
MAX_NAME_LEN = 40

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

def html_escape2(text):
    return html_escape(html_escape(text))

def getitem(seq, i, default=None):
    return i in range(len(seq)) and seq[i] or default

def accmulate(func, seq, initial):
    def handle(x, i):
        x.append(func(x[-1], i))
        return x
    return reduce(handle, seq, [initial])

def popen(cmd, input=None):
    return Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT).communicate(input=input)[0]

def last_n_part_of_path(path, n):
    return '/'.join(path.split('/')[-n:])

def remove_common_prefix(ls, maxdepth=2):
    prefix = re.sub('/[^/]*$', '/', os.path.commonprefix(ls))
    return [last_n_part_of_path(i[len(prefix):], maxdepth) for i in ls]

def addr2name(addrs, exe, maxdepth=2):
    def addr2line(exe, addrs):
        return popen('addr2line -f -C -e %s'%(exe), '\n'.join('0x%x'%i for i in addrs))
    fn_trace = re.findall('^(.+)\n(.+)$', addr2line(exe, addrs), re.M)
    funcs, lineno = zip(*fn_trace)
    return dict(zip(addrs, zip(funcs, remove_common_prefix(lineno, maxdepth))))

def call_chains(trace):
    def call_link((_enter_exit, parent), (enter_exit, func, lineno)):
        return enter_exit, enter_exit == 'E' and (parent, (func, lineno)) or parent[0]
    return [chain for enter_exit, chain in accmulate(call_link, trace, (None, (None,None))) if enter_exit == 'E']

def safe_atoi(str, default=0):
    try:
        return int(str)
    except Exception:
        return default
    
def resolve_counter_names(counters, exe, maxdepth=2):
    name_map = addr2name([callee for (caller, callee) in counters], exe, maxdepth)
    topn = safe_atoi(os.getenv('TOPN'), 100)
    return dict(((caller, callee), (count, name_map[caller], name_map[callee])) for (caller, callee), count in counters.most_common(topn) if caller)

def trace2plain(counters, exe, maxdepth=2):
    def func_repr(func):
        return func and '@'.join(func)
    return '\n'.join('%d %s => %s'%(count, func_repr(caller), func_repr(callee)) for (_caller, _callee), (count, caller, callee) in resolve_counter_names(counters, exe, maxdepth).items())

def trace2dot(trace, exe, maxdepth=2):
    def func_repr(func):
        return func and '@'.join(func)
    def call_link2dot(_caller, _callee, caller, callee, count):
        edge = '''x%(_caller)d->x%(_callee)s[label="%(count)d", fontsize=%(fontsize)d, penwidth=%(linewidth)d];
 x%(_callee)d[shape=plaintext, style=filled, bgcolor=lightgray,
   label=< <font face="monospace" color="black" point-size="%(fontsize)d">%(escaped_func)s</font>
   <br/><font color="navy" point-size="%(smallfontsize)d">%(escaped_lineno)s</font> >];'''
        func, lineno = callee
        func, lineno = func[:MAX_NAME_LEN], lineno[:2*MAX_NAME_LEN]
        return edge %dict(_caller=_caller, _callee=_callee,
                          count=count, escaped_func=html_escape2(func), escaped_lineno=html_escape2(lineno), 
                          fontsize=5*math.log(3*count), smallfontsize=2*math.log(3*count), linewidth=math.log(3*count))
    digraph = '''
digraph G {
node[shape=box, style=filled];
rankdir=LR;
%s
};'''
    return digraph% '\n'.join(call_link2dot(_caller, _callee, caller, callee, count) for (_caller, _callee),(count, caller, callee) in resolve_counter_names(trace, exe, maxdepth).items())

def create_trace_so(dir):
    write('%s/trace.c'%(dir), trace_so_source)
    return call('gcc -fPIC -rdynamic --shared %(dir)s/trace.c -o %(dir)s/trace.so'%dict(dir=dir), shell=True) == 0

def traced_call(trace_file, trace_so, args):
    os.putenv('LD_PRELOAD', '%s/trace.so'%(cwd))
    os.putenv('TRACE', trace_file)
    p = Popen(args)
    os.putenv('LD_PRELOAD', '')
    return p

def get_trace_counters(file):
    stack, counter = array.array('L', [0]), Counter()
    with open(file) as f:
        try:
            while f:
                stack.fromfile(f, 1)
                if stack[-1] == 0:
                    break
                if stack[-1] == 0xFFFFFFFFFFFFFFFF:
                    stack.pop()
                    stack.pop()
                else:
                    counter.update([(stack[-2], stack[-1])])
        except EOFError:
            pass
    return counter

def trace_term_plain(counters, exefile, outfile):
    write(outfile, trace2plain(counters, exefile))

def trace_term_dot(counters, exefile, outfile):
    write(outfile, trace2dot(counters, exefile))
    
def trace_term_png(counters, exefile, outfile):
    print popen('dot -Tpng -o %s'% outfile, input=trace2dot(counters, exefile))

def trace_term_svg(counters, exefile, outfile):
    print popen('dot -Tsvg -o %s'% outfile, input=trace2dot(counters, exefile))
    
def create_trace_file(trace_file, trace_so, args):
    p = traced_call('%s.trace'% trace_basename, '%s/trace.so'% cwd, sys.argv[1:])
    signal.signal(signal.SIGINT, lambda signo, frame: os.kill(p.pid, signal.SIGINT))
    p.wait()
    return os.path.exists(trace_file)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda signo, frame: err('caught SIGINT.\n'))
    trace_output = os.getenv('TRACE') or (len(sys.argv) > 1 and os.path.basename(sys.argv[1]) + '.png') or err(__doc__)
    trace_basename, trace_ext = os.path.splitext(trace_output)
    trace_term = trace_ext and trace_ext[1:]
    cwd = os.path.dirname(os.path.realpath(__file__))
    os.path.exists('%s/trace.so'%(cwd)) or create_trace_so(cwd) or err('create "%s/trace.so" failed.\n'%(cwd))
    sys.argv[1:] and (create_trace_file('%s.trace'% trace_basename, '%s/trace.so'% cwd, sys.argv[1:]) or err('create "%s.trace" failed.\n'% trace_basename))
    exefile = os.getenv('EXEFILE') or (len(sys.argv) > 1 and sys.argv[1] or err('not provide exefile to resolve func names.\n'))
    counters = get_trace_counters('%s.trace'% trace_basename) or err('not valid "%s.trace".\n'% trace_basename)
    _term = globals().get('trace_term_%s'%(trace_term)) or (lambda counters, exefile, outfile: err('no term defined for %s.'% trace_term))
    _term(counters, exefile, trace_output)
