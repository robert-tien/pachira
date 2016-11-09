# -*- coding: utf-8 -*-
import os
import numpy as np
import re
import itertools
from collections import Counter
import pdb
import removestopwords as rmsw
import datetime
import logging

#global debug flag
DBG = 0
PATH="/home/robert_tien/work/word2vec/"
CUR_DIR="/home/robert_tien/work/pachira/cnn-text-classification-tf/"

def read_stopwords(fn):
    #fn = "../zh_stopwords.json" 
    s = open(fn,"r").readline()
    l = s.split(",")
    return l


def clean_str(string):
    #return removestopwords.remove_stopwords(string)
    
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    """
    string = re.sub(r"嗯", "", string)
    string = re.sub(r"喂", "", string)
    string = re.sub(r"啊", "", string)
    string = re.sub(r" 对 ", " ", string)
    string = re.sub(r"吧", "", string)
    string = re.sub(r"哎", "", string)
    string = re.sub(r"那 ", "", string)
    string = re.sub(r"哦", "", string)
    string = re.sub(r"呃", "", string)
    string = re.sub(r"啦", "", string)
    string = re.sub(r"啥", "", string)
    string = re.sub(r"呀", "", string)
    string = re.sub(r"咋", "", string)
    string = re.sub(r"呢", "", string)
    string = re.sub(r"咧", "", string)
    string = re.sub(r"咯", "", string)
    string = re.sub(r"也", "", string)
    string = re.sub(r"咱", "", string)
    string = re.sub(r"  行 ", " ", string)
    string = re.sub(r"坐席 :", "", string)
    string = re.sub(r"客户 :", "", string)
    string = re.sub(r"嘛", "", string)
    return string.strip().lower()

def readdir(dir):
    list = []
    fnlist = []
    i = 0
    maxlen = 0
#dir = PATH+"chinesetokenization/busroles_txt1/cat1_out/"
    for filename in os.listdir(dir):
    #logging.info(filename)
        #pdb.set_trace()
        f = open(dir+"/"+filename,"r")
        fn = f.readline()
        fname = dir+"/"+filename
        s = f.read()
        if len(s.split(" ")) > maxlen:
           maxlen = len(s.split(" "))
           maxfn = filename
        if len(s) < 10: #24: 
            logging.info( "skip "+filename+": too short\n")
            continue
        if len(s) > 1000000:
            logging.info("skip "+filename+": too long\n")
            continue # skip too big files for now
        else:
            i+=1
            #print i , len(s)
            list.append(s)
            fnlist.append(fname) # remember its filename
    logging.info( maxfn+" "+str( maxlen))
    return list, fnlist

def load_data_and_labels_bootstrap_eval():
    # Load data from files
    #pdb.set_trace()
    logging.info("Loading cat1...")
    dir =CUR_DIR+"data/my_classification/all_unclass"
    cat1_examples, fnlist = readdir(dir)
    cat1_examples = [s.strip() for s in cat1_examples]
    logging.info("cat1_examples "+str(len(cat1_examples)))

    x_text = cat1_examples 
    x_text = [rmsw.remove_useless(clean_str(sent)) for sent in x_text]
    logging.info( "x_text "+str(len(x_text)))
    #pdb.set_trace()
    # Generate labels
    cat1_labels = [0 for _ in cat1_examples]
    y = cat1_labels
    return [x_text, y, fnlist]

def load_data_and_labels_bootstrap(kind):
    # Load data from files
    #pdb.set_trace()
    logging.info("Loading "+kind+" ...")
    dir = CUR_DIR+"data/my_classification/class/"+kind
    cat1_examples,_ = readdir(dir)
    cat1_examples = [s.strip() for s in cat1_examples]
    logging.info("cat1_examples "+str(len(cat1_examples)))

    logging.info("Loading "+kind+" ...")
    dir = CUR_DIR+"data/my_classification/allothers"+"_"+kind
    cat2_examples,_ = readdir(dir)
    cat2_examples = [s.strip() for s in cat2_examples]
    logging.info("cat2_examples "+str(len(cat2_examples)))

    x_text = cat1_examples + cat2_examples
    x_text = [clean_str(sent) for sent in x_text]
    logging.info("x_text "+str(len(x_text)))
    #pdb.set_trace()
    # Generate labels
    cat1_labels = [[0, 1] for _ in cat1_examples]
    cat2_labels = [[1, 0] for _ in cat2_examples]
    y = np.concatenate([cat1_labels, cat2_labels], 0)
    return [x_text, y]
 
def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    logging.info("Loading cat1...")
    dir = PATH+"chinesetokenization/busroles_txt1/cat1_out/"
    cat1_examples,_ = readdir(dir)
    cat1_examples = [s.strip() for s in cat1_examples]
    logging.info("cat1_examples "+str(len(cat1_examples)))
    logging.info("Loading cat2...")
    dir = PATH+"chinesetokenization/busroles_txt1/cat2_out/"
    cat2_examples,_ = readdir(dir)
    cat2_examples = [s.strip() for s in cat2_examples]
    logging.info("cat2_examples "+str(len(cat2_examples)))
    logging.info("Loading cat3...")
    dir = PATH+"chinesetokenization/busroles_txt1/cat3_out/"
    cat3_examples,_ = readdir(dir)
    cat3_examples = [s.strip() for s in cat3_examples]
    logging.info("cat3_examples "+str(len(cat3_examples)))
    logging.info("Loading cat4...")
    dir = PATH+"chinesetokenization/busroles_txt1/cat4_out/"
    cat4_examples,_ = readdir(dir)
    cat4_examples = [s.strip() for s in cat4_examples]
    logging.info("cat4_examples "+str(len(cat4_examples)))
    logging.info("Loading cat5...")
    dir = PATH+"chinesetokenization/busroles_txt1/cat5_out/"
    cat5_examples,_ = readdir(dir)
    cat5_examples = [s.strip() for s in cat5_examples]
    logging.info("cat5_examples "+str(len(cat5_examples)))
# cat all and clean 
    x_text = cat1_examples + cat2_examples + cat3_examples + cat4_examples + cat5_examples
    x_text = [clean_str(sent) for sent in x_text]
    logging.info("x_text "+str(len(x_text)))
    #pdb.set_trace()
    # Generate labels
    cat1_labels = [[0, 0, 0, 0, 1] for _ in cat1_examples]
    cat2_labels = [[0, 0, 0, 1, 0] for _ in cat2_examples]
    cat3_labels = [[0, 0, 1, 0, 0] for _ in cat3_examples]
    cat4_labels = [[0, 1, 0, 0, 0] for _ in cat4_examples]
    cat5_labels = [[1, 0, 0, 0, 0] for _ in cat5_examples]
    y = np.concatenate([cat1_labels, cat2_labels, cat3_labels, cat4_labels, cat5_labels], 0)
    """
    cat1_labels = [[0, 1] for _ in cat1_examples]
    cat2_labels = [[1, 0] for _ in cat2_examples]
    y = np.concatenate([cat1_labels, cat2_labels], 0)
    """
    return [x_text, y]

"""
def len_stat(x_text):
    res = []
    for sent in x_text:
        res.append(len(list(sent.split())
    return res
"""

def build_w2v_dict():
  # Load wiki.zh.vector file
  fn = PATH+"chinesetokenization/msr_mecab_test/wiki.zh.text.vector"
  logging.info("Loading "+fn)
  dictionary = dict()
  f = open(fn, "r")
  vects = f.readlines()
  for vect in vects:
    l = vect.split()
    dictionary[l[0]] = np.array(map(float,l[1:]))
  return dictionary

def build_dict(x_text):
    dictionary = dict()
    i = 0
    for sent in x_text:
      l = sent.split()
      for w in l:
        if w not in dictionary:
            i += 1
            dictionary[w]=i
    dictionary["<UNK>"]=0
    return dictionary

def fit_transform(x_text, dictionary, max):
    res = []
    for sent in x_text:
      list = []
      i = 0
      l = sent.split()
      for w in l:
        list.append(dictionary[w])
        i += 1
      for j in range(max - i):
        list.append(0)
      #pdb.set_trace()
      res.append(list)    
    return res

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    if DBG > 0:
        pdb.set_trace()
    data = np.array(data)
    data_size = len(data)
    #num_batches_per_epoch = int(len(data)/batch_size) + 1
    num_batches_per_epoch = int(len(data)/batch_size)
    if len(data)%batch_size != 0:
        num_batches_per_epoch += 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if start_index==end_index:
                pdb.set_trace()
            if DBG > 0:
                logging.info("start_index:"+str(start_index)+" end_index:"+str(end_index))
            yield shuffled_data[start_index:end_index]
