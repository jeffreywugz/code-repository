#!/usr/bin/python2
import sys, os
import re, string
import math
from subprocess import Popen, PIPE, STDOUT
import pygraphviz
import numpy as npa
import matplotlib.pyplot as plt

def popen(cmd):
    return Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT).communicate()[0]

def get_packages():
    return popen("pacman -Ql|awk '{print $1}'|uniq").strip().split('\n')

def get_explict_installed_packages():
    return popen("pacman -Qeq").strip().split()

def package_info(p):
    def get_list(content, tag):
        m = re.search('%s\s+:\s+(.*)'%tag, content)
        list = (m and m.group(1) or '').split()
        return filter(lambda x: x != 'None', [re.split('[<>=]', p)[0] for p in list])
    def get_installed_size(content):
        return float(re.search("Installed Size\s*:\s*([0-9.]+)", content).group(1))
    i = popen("pacman -Qi %s"% p)
    return get_list(i, 'Groups'), get_list(i, 'Provides'), get_list(i, 'Depends On'), get_installed_size(i)

def deps_graph():
    G = pygraphviz.AGraph(directed=True)
    all_packages, explict_installed_packages = get_packages(), get_explict_installed_packages()
    for n in explict_installed_packages:
        G.add_node(n, shape='plaintext', fontcolor='red')
    all_provides = []
    for p in all_packages:
        groups, provides, deps, size = package_info(p)
        all_provides.extend(provides)
        [G.add_edge(i, p, color='grey') for i in provides]
        [G.add_edge(p, i, color='blue') for i in deps]
        [G.add_edge(i, p, color='yellow') for i in groups]
        [G.get_node(i).attr.update(shape='box', fontsize=30) for i in groups]
        G.add_node(p) # deal with orphans
        G.get_node(p).attr.update(label="%s\\n%dK"%(p, size), shape='plaintext', color='black', fontsize=math.log(size,1.5))
    all_provides = list(set(all_provides))
    for n in all_provides:
        G.get_node(n).attr.update(shape='plaintext', fontcolor='grey')
    for n in all_provides:
        if G.has_node(n) and not G.in_edges(n):
            G.remove_node(n)
    return G

def pkgs_size():
    return [package_info(p)[3] for p in get_packages()]

def draw_deps_graph():
    G = deps_graph()
    G.write('pkg-deps.dot')
    G.draw('pkg-deps.png', prog='dot')

def hist_plot_pkgs_size():
    plt.hist(pkgs_size(), bins=20, log=True)
    plt.xlabel('size(K)')
    plt.ylabel('count')
    plt.savefig('pkg-sizes.png')
    
if __name__ == '__main__':
    draw_deps_graph()
    # hist_plot_pkgs_size()
    
