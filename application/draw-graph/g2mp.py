#!/usr/bin/env python
from string import Template as TPL

size_tpl=TPL('xlen=$xlen; ylen=$ylen;')
vertex_tpl=TPL('circleit.$v("$v"); $v.c=($vx*xlen, $vy*ylen); drawboxed($v);')
edge_tpl=TPL('join($ea, $eb);')
graph_tpl=TPL(r"""
input boxes
prologues:=1;
\beginfig(1);
vardef join(suffix a,b)=
	drawarrow a.c--b.c cutbefore bpath.a cutafter bpath.b;
enddef;
$size
$vertexs
$edges
endfig;
end
""")

def g2mp(g):
    size=size_tpl.substitute(xlen=g.xlen, ylen=g.ylen)
    vertexs='\n'.join(
            map(lambda v:vertex_tpl.substitute(v=v[0],vx=v[1][0],vy=v[1][1]),
                g.vertexs.items()))
    edges='\n'.join(
            map(lambda e:edge_tpl.substitute(ea=e[0],eb=e[1]),
                g.edges))
    return graph_tpl.substitute(size=size, vertexs=vertexs, edges=edges)

