#!/usr/bin/env python

from config import *

def get_line(fcboard,xys):
    L=[]
    for x,y in xys:
        if x>=0 and x<XCOUNTS and y>=0 and y<YCOUNTS:
            L.append(fcboard[x][y])
        else:
            L.append(forbidden)
    return L;

def get_fclists(fcboard,x,y):
    L1=get_line(fcboard,zip(range(x-4,x+5),[y]*9))
    L2=get_line(fcboard,zip([x]*9,range(y-4,y+5)))
    L3=get_line(fcboard,zip(range(x-4,x+5),range(y+4,y-5,-1)))
    L4=get_line(fcboard,zip(range(x-4,x+5),range(y-4,y+5)))
    return L1,L2,L3,L4


