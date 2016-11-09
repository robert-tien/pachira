# -*- coding: utf-8 -*-
import os
import pdb
#import itertools
from operator import add
import numpy as np

dbg = 0.0  

def build_wordvec_dict(fn):
    #fn = "../../../word2vec/chinesetokenization/msr_mecab_test/wiki.zh.text.vector.40"
    #print filename
    #pdb.set_trace()
    dictionary=dict()
    #pdb.set_trace()
    s = open(fn,"r").readlines()
    for word in s:
        l=word.split()
        dictionary[l[0]]=np.array(l[1:], dtype=np.float32)
    return dictionary

def save_dict(dict, name):
    json.dump(d,open(name,'w'))
    return

"""
  We bypass the embedding layer since we already have wordvec generated. We directly construct
  a tensor of shape (?,seqlen,embedding_size). This tensor will go to the convulation layer.
  The feed_dict for needs to get batch of these tensor of the same shape.
  Here we are generate the doctensor of shape(seqlen,size)  by averaging/max-ing every m 
  (defined below in code) wordvecs component-wise with the last vector(row) averaging/max-ing
  len(ls)-(seqlen-1)*m.
  By partitioning the doc into sections where each section is represented by a wordvec-
  derived vector will keep some degree of word ordering and allow convultion layer later to
  derive/learn features in the document for classification and other usage.
  Note: max-ing scheme is not yet implemented.
"""
def build_doctensor(dictionary, string, size=128, seqlen=56, ave=True, max=False):
    if dbg > 1.0 : print "build_doctensor start"
    #pdb.set_trace()
    res = np.full([seqlen,size],0.0) 
    i = j = 0 
    if dbg > 2.0 : print "i="+str(i)
    if dbg > 2.0 : print "j="+str(j)
    ls = string.split() # in chinese string is a phrase(词) 分词的结果
    m = len(ls)//seqlen
    r = len(ls)%seqlen
    if dbg > 1.0 : print "len:"+str(len(ls))+" m= "+str(m)+" r="+str(r)
    if r != 0:
      m += 1
    for w in ls:
      i += 1
      if dbg > 3.0: print "i="+str(i)
      if w in dictionary:
        arr = dictionary[w]
      else:
        if dbg > 1.0 : print "key "+w+" is not in dictionary"
        arr = np.zeros(size)
        #continue
      if (ave):
        a0 = res[j][0]
        res[j] = res[j] + arr
        assert(res[j][0]==arr[0]+a0)
        if i==len(ls) and r > 0 :
          res[j] = res[j]/r
          break
        if i%m==0 :
          #pdb.set_trace()
          res[j] = res[j]/m
          j += 1
          if dbg > 3.0 : print "j="+str(j)
    if dbg > 1.0 : print "build_doctensor end"
    return res
# seqlen=3, len(ls)=14, m=4+1 
# res[3,size]
    
def fit_transform(x_text, dictionary):
    res = []
    if dbg > 0.8: pdb.set_trace()
    for sent in x_text:
      tensor = build_doctensor(dictionary, sent)
      res.append(tensor)    
    if dbg > 0.8: pdb.set_trace()
    return res


#def batch_iter:
