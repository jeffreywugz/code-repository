#!/usr/bin/env python

import os
import Jigsaw

def pySolve(state):
    jigsaw=Jigsaw.Jigsaw("Jigsaw",None,tuple(state),0)
    if not jigsaw.canSolve():
        return False,None
    else:
        return True,Jigsaw.search(jigsaw)

jigsawSolver='./jigsaw'
def cppSolve(state):
    w,r=os.popen2(jigsawSolver)
    w.write(reduce(lambda a,b: a+' '+str(b),state,''))
    w.close()
    if r.readline()=='no answer!\n':
        return False,None
    else:
        return True,[[int(i) for i in s.split()] for s in r.readlines()]

if __name__ == '__main__':
    initialState=(
            1,2,3,4,
            5,6,7,8,
            9,10,11,12,
            13,14,15,16
        )
    print cppSolve(initialState)


