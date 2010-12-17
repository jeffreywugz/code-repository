#!/usr/bin/python
import heapq
import copy

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

def astar_search(start, is_target):
    open_nodes, closed_nodes = [start], set()
    while open_nodes:
        node = heapq.heappop(open_nodes)
        if is_target(node): return node.path()
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
    print astar_search(start, lambda x: x == target)
    
class BtNodeR:
    def __init__(self):
        pass
    
    def is_leaf(self):
        return True

    def get_children(self):
        return []

    def is_compatible(self):
        return False

    def get_state(self):
        return None
    
    def __repr__(self):
        return repr(self.get_state())

def _bt_search_recursive(root, result):
    if not root.is_compatible():
        return
    if root.is_leaf():
        result.append(copy.copy(root.get_state()))
        return
    for c in root.get_children():
        _bt_search_recursive(c, result)

def bt_search_recursive(root):
    result = []
    _bt_search_recursive(root, result)
    return result

class BtNode:
    def __init__(self):
        pass

    def is_last_node(self):
        return True

    def is_last_sibling(self):
        return True

    def is_leaf(self):
        pass
    
    def next_sibling(self):
        pass

    def next_uncle(self):
        pass

    def first_child(self):
        pass
    
    def get_state(self):
        return None

    def __repr__(self):
        return repr(self.get_state())
    
def bt_search(root):
    """we need fake `last_sibling' and `last_node' and `last_level_leaf'
    so node.is_last_node() and node.is_last_sibling() and node.first_child() could work.
    """
    result = []
    while not root.is_last_node():
        if root.is_leaf():
            result.append(copy.copy(root.get_state()))
        if root.is_leaf() or root.is_last_sibling():
            root.next_uncle()
        elif not root.is_compatible():
            root.next_sibling()
        else:
            root.first_child()
    return result
        
class TestBtNodeR(BtNodeR):
    def __init__(self, n):
        self.n, self.d = n, 0
        self.comb = [0] * n
        BtNodeR.__init__(self)

    def get_children(self):
        self.d += 1
        for i in range(self.n):
            self.comb[self.d - 1] = i
            yield self
        self.d -= 1
            
    def is_leaf(self):
        return self.d == self.n

    def is_compatible(self):
        def _compatible(x, y, ix, iy):
            return x !=y and x-y != ix-iy and x-y != iy - ix
        return all([_compatible(self.comb[i], self.comb[self.d-1], i, self.d-1) for i in range(self.d-1)])

    def get_state(self):
        return self.comb[:self.d]

class TestBtNode(BtNode):
    def __init__(self, n):
        self.n, self.d = n, 0
        self.comb = [0] * n
        BtNode.__init__(self)

    def is_leaf(self):
        return self.d == self.n+1

    def is_last_sibling(self):
        return self.comb[self.d-1] == self.n

    def is_last_node(self):
        return self.d < 0
    
    def next_sibling(self):
        self.comb[self.d-1] += 1

    def next_uncle(self):
        self.d -= 1
        if self.d > 0:
            self.comb[self.d-1] += 1

    def first_child(self):
        self.d += 1
        if self.d <= self.n:
            self.comb[self.d-1] = 0

    def get_state(self):
        return self.comb[:self.d]

    def is_compatible(self):
        def _compatible(x, y, ix, iy):
            return x !=y and x-y != ix-iy and x-y != iy - ix
        return all([_compatible(self.comb[i], self.comb[self.d-1], i, self.d-1) for i in range(self.d-1)])

def test_bt_search_r():
    print len(bt_search_recursive(TestBtNodeR(12)))

def test_bt_search():
    print len(bt_search(TestBtNode(12)))
    
# print "test_astar_search()"
# test_astar_search()
# print "test_bt_search_r()"
# test_bt_search_r()
print "test_bt_search()"
test_bt_search()
