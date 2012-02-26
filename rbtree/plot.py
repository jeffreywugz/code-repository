#!/usr/bin/env python2
import re
import sys
from itertools import count

counter = count(0)
def get_node_def(value):
    node_id = counter.next()
    node_def = '%d[style=filled, fillcolor=lightgray, label="%s"]' %(node_id, str(value))
    return node_id, node_def

def dump_rb_tree(name, root):
    def _dump_rb_tree(parent, root):
        if type(root) != list and type(root) != tuple:
            raise Exception("wrong type: %s"% type(root))
        if not root:
            nil_id, nil_def = get_node_def("nil")
            return '%s; %d->%d;\n'% (nil_def, parent, nil_id)
        (color, value), left, right = root
        root_id, root_def = get_node_def(value)
        return '%s;%d->%d[color=%s];\n%s%s' %(root_def, parent, root_id, color, _dump_rb_tree(root_id, left), _dump_rb_tree(root_id, right))

    root_id, root_def = get_node_def(name)
    return 'subgraph {%s;\n %s\n}' % (root_def, _dump_rb_tree(root_id, root))

def parse_tree(tree_repr):
    sys.stderr.write(tree_repr)
    return eval(tree_repr, dict(Nil=(), R='red', B='Black'))[0]

if __name__ == '__main__':
    tree_reprs = re.findall('^-+rb_tree_begin-+$(.*?)^-+rb_tree_end-+$', sys.stdin.read(), re.S|re.M)
    dot_file = '\n'.join(dump_rb_tree("Tree%d"% i, parse_tree(tree)) for i, tree in enumerate(tree_reprs))
    print 'digraph G { %s }' % dot_file
