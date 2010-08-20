#!/usr/bin/python

import sys
from pymmseg import mmseg  
   
mmseg.dict_load_defaults()  
text = open(sys.argv[1]).read()
algor = mmseg.Algorithm(text)  
for tok in algor:
    print tok.text

