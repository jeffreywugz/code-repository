#!/bin/env python

def search(start, is_target):
    open_nodes, close_nodes = [start], []
    while open_nodes:
        expanding_node = open_nodes.pop(0)
        if expanding_node in close_nodes:
                continue
        if is_target(expanding_node):
            break
        children = expanding_node.gen_children()
        open_nodes.extend(children)
        open_nodes.sort(key = lambda x: x.cost)
        close_nodes.append(expanding_node)
    else:
        return None
    
    path = []
    while expanding_node:
        path.insert(0, expanding_node)
        expanding_node = expanding_node.parent
    return path

    
class CupNode:
    def __init__(self, parent, cost, c1, c2, C1, C2):
        self.parent, self.cost = parent, cost
        self.c1, self.c2, self.C1, self.C2 = c1, c2, C1, C2

    def _new_child(self, c1,  c2):
        return CupNode(self, self.cost+1, c1, c2, self.C1, self.C2)

    def gen_children(self):
        child1 = self._new_child(0, self.c2)
        child2 = self._new_child(self.c1, 0)
        child3 = self._new_child(self.C1, self.c2)
        child4 = self._new_child(self.c1, self.C2)

        if self.C1 >= self.c1 + self.c2:
            c1, c2 = self.c1 + self.c2, 0
        else:
            c1, c2 = self.C1, self.c1 + self.c2 - self.C1
        child5 = self._new_child(c1, c2)
        
        if self.C2 >= self.c1 + self.c2:
            c1, c2 = 0, self.c1 + self.c2
        else:
            c1, c2 = self.c1 + self.c2 - self.C2, self.C2
        child6 = self._new_child(c1, c2)
        
        return [child1, child2, child3, child4, child5, child6]

    def __eq__(self, node):
        return (self.c1, self.c2) == (node.c1, node.c2)

    def __repr__(self):
        return "<%d %d>"%(self.c1, self.c2)

if __name__ == '__main__':
    print search(CupNode(None, 0, 0, 0, 5, 6), lambda x: x.c1==4 or x.c2 == 4)
