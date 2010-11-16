#!/usr/bin/python
import heapq

class Node:
    def __init__(self, parent):
        self.parent = parent

    def path(self):
        p = []
        cur_node = self
        while cur_node:
            p.append(cur_node)
            cur_node = cur_node.parent
        return list(reversed(p))
        
    def get_state(self):
        return None

    def __eq__(self, x):
        return self.get_state() == x.get_state()
        
    def __cmp__(self, x):
        return cmp(self.cost(), x.cost())

    def __hash__(self):
        return hash(self.get_state())
    
    def __repr__(self):
        return repr(self.get_state())
    
    def get_children(self):
        return []

    def cost(self):
        return 0

def astar_search(start, target):
    open_nodes, closed_nodes = [start], set()
    while open_nodes:
        node = heapq.heappop(open_nodes)
        if node == target: return node.path()
        if node in closed_nodes: continue
        closed_nodes.add(node)
        for i in node.get_children():
            heapq.heappush(open_nodes, i)
    return None

class TestNode(Node):
    def __init__(self, parent, n):
        Node.__init__(self, parent)
        self.n = n

    def get_state(self):
        return self.n

    def get_children(self):
        return [TestNode(self, self.n*i) for i in [2, 3, 5]] + [TestNode(self, self.n+1)]

    def cost(self):
        if not self.parent: return 0
        else: return self.parent.cost() + 1

def test_astar_search():
    start, target = TestNode(None, 0), TestNode(None, 12345)
    print astar_search(start, target)
    
print "test_astar_search()"
test_astar_search()
