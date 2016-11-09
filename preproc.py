#! /usr/bin/env python

import pdb
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
# Load data
print("Loading data...")
#pdb.set_trace()
x_text, y = data_helpers.load_data_and_labels()

dictionary = data_helpers.build_dict(x_text) 
res = []
for sent in x_text:
   res.append(len(list(sent.split())))
res.sort()
print res

