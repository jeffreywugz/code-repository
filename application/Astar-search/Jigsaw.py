#!/usr/bin/env python
move=(
    (1,4), (1,-1,4), (1,-1,4), (-1,4),
    (1,4,-4), (1,-1,4,-4), (1,-1,4,-4), (-1,4,-4),
    (1,4,-4), (1,-1,4,-4), (1,-1,4,-4), (-1,4,-4),
    (1,-4), (1,-1,-4), (1,-1,-4), (-1,-4),
)
tab=(
    0,1,0,1,
    1,0,1,0,
    0,1,0,1,
    1,0,1,0
)

class Jigsaw:
    def __init__(self,name,parent,state,cost):
        self.name=name
        self.parent=parent
        self.state=state
        for i in range(len(self.state)):
            if self.state[i]==16:
                self.pos=i
                break
        self.cost=cost

    def canSolve(self):
        reverse=0
        for i in range(len(self.state)):
            for j in range(i+1,len(self.state)):
                if self.state[i]>self.state[j]:reverse+=1
        return (reverse+tab[self.pos])%2==0

    def isAnswer(self):
        return self.restCost()==0

    def children(self):
        chs=[]
        for i in move[self.pos]:
            p0,p1=self.pos,self.pos+i
            state=list(self.state)
            state[p0],state[p1]=state[p1],state[p0]
            chs.append(Jigsaw(self.name,self,tuple(state),self.cost+1))
        return chs

    def allCost(self):
        return self.cost+self.restCost()

    def restCost(self):
        rCost=0
        for i in range(len(self.state)):
            if self.state[i]!=16 and self.state[i]!=i+1:
                rCost+=1
        return rCost

    def __str__(self):
        return str(self.state)

def search(root):
    open=[root]
    nodes={}
    while open:
        enode=open.pop(0)
        nodes[enode.state]=True
        if enode.isAnswer():break
        open.extend(filter(lambda node:not nodes.has_key(node.state),
            enode.children()))
        open.sort(key=lambda node:node.allCost())
    else:
        return None
    path=[]
    parent=enode
    while parent:
        path.insert(0,parent)
        parent=parent.parent
    return path

if __name__ == '__main__':
    cannotSolveinitialState=(
        1,3,4,5,
        2,16,5,12,
        7,6,11,14,
        8,9,10,13
    )
    canSolveinitialState=(
           1,2,3,4,
           16,6,7,8,
           5,9,10,12,
           13,14,11,15
           )
    jigsaw=Jigsaw("Jigsaw",None,canSolveinitialState,0)
    if not jigsaw.canSolve():
        print "can't solve!"
    else:
        for i in search(jigsaw):
            print i
