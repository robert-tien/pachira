import os
import pdb

list = []
i = 0
dir = "../../../word2vec/chinesetokenization/busroles_txt1/cat1_out/"
for filename in os.listdir(dir):
    #print filename
    #pdb.set_trace()
    f = open(dir+"/"+filename,"r")
    f.readline()
    s = f.read()
    if len(s) < 1024: 
        continue
    else:
        i+=1
        print i , len(s)
        list.append(s)

#pdb.set_trace()
