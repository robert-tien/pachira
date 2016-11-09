# -*- coding: utf-8 -*-
import pdb
import os
import re

def read_stopwords(fn):
    #fn = "../zh_stopwords.json" 
    s = open(fn,"r").readline()
    l = s.split(",")
    return l

def remove_stopwords(stopwords,string):
    for w in stopwords:
        r0=re.compile(w)
        string = r0.sub("",string)
    return string

def gen_sub_stmt(stopwords):
    for w in stopwords:
        print "      string = re.sub(r"+w+", \"\", string)"

l = read_stopwords("../zh_stopwords.json")
gen_sub_stmt(l)



"""
>>> re.compile('、')
<_sre.SRE_Pattern object at 0x7f29150574c8>
>>> r0=re.compile('、')
>>> r0.sub("","this is 、test !!")
'this is test !!'
"""
