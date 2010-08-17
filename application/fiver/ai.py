#!/usr/bin/env python

from config import *
import util
import ai_data

def get_weight(L,color):
    k=0
    for i in range(5):
        if L[i]==color:
            k+=pow(2,i)
        elif L[i]!=none:
            return 0
    return ai_data.weight[k]

def get_line_value(L,color):
    v=0
    for i in range(5):
        v+=get_weight(L[i:i+5],color)
    return v

def get_color_value(fcboard,x,y,color):
    fcboard[x][y]=color
    v=0
    for L in util.get_fclists(fcboard,x,y):
        v+=get_line_value(L,color)
    fcboard[x][y]=none
    return v

def get_value(fcboard,x,y):
    if fcboard[x][y]!=none:
        return -1
    return get_color_value(fcboard,x,y,white)+get_color_value(fcboard,x,y,black)

def get_move(fcboard):
    m,mx,my=0,-1,-1
    for x in range(XCOUNTS):
        for y in range(YCOUNTS):
            tm=get_value(fcboard,x,y)
            if tm>=m:
                m,mx,my=tm,x,y
    return mx,my
