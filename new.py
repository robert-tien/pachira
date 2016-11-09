#! /usr/bin/env python

import numpy as np
import re
import os
import time
import datetime

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
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
    return string.strip().lower()

def test_py():
    positive_examples = list(open("./test.txt", "r").readlines())
    negative_examples = list(open("./test1.txt", "r").readlines())
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
#    return [x_text, y]    
    return [x_text, positive_labels, negative_labels, y]    
    
# Load data
print("Loading data...")
x_text, pos, neg, y = test_py()
print("x_text")
print x_text
print("pos")
print pos
print("neg")
print neg
print("y")
print y
