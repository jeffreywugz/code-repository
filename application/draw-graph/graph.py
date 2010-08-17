#!/usr/bin/env python
import g2mp
import layout

class Graph:
    def __init__(self, fname):
        self.xlen, self.ylen = 600, 400
        self.edges=map(lambda e:e.split(), open(fname).readlines())

    def layout(self):
        vertexs={}
        def add_vertex(v):
            if not vertexs.has_key(v):
                vertexs[v]=add_vertex.index
                add_vertex.index+=1
        add_vertex.index=0
        for a, b in self.edges:
            add_vertex(a)
            add_vertex(b)

        edge_seq=reduce(lambda seq,e:seq.extend(e) or seq,
                map(lambda e:(vertexs[e[0]],vertexs[e[1]]), self.edges),[])
        self.vertexs={}
        pos_seq=layout.layout(edge_seq)
        for v,i in vertexs.items():
            self.vertexs[v]=(pos_seq[i*2], pos_seq[i*2+1])

    def write(self, fname):
        open(fname,'w').write(g2mp.g2mp(self))

if __name__ == '__main__':
    import sys
    g=Graph(sys.argv[1])
    g.layout()
    g.write(sys.argv[2])

