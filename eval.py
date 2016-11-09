#! /usr/bin/env python
# Usage: python eval.sh bootstrap eval.$NOW.log checkpt_dir eval.$NOW.out.sh dir_training
# where dir_training is $2 for run_traineval.sh
import pdb, sys, os
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import readwordvec


# Parameters
# ==================================================
PATH="/home/robert_tien/work/word2vec/"
OUT_DIR="/home/robert_tien/work/pachira/cnn-text-classification-tf/data/runs/"
if len(sys.argv) > 2:
    f = open(OUT_DIR+sys.argv[3]+"/"+sys.argv[2],"w")
if len(sys.argv) < 5 and len(sys.argv) > 1:
    print("Usage: python eval.sh bootstrap out_dir_name checkpt_dir\n")
    f.write("Usage: python eval.sh bootstrap out_dir_name checkpt_dir outfile\n")
    sys.exit()
# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# seed run
#tf.flags.DEFINE_string("checkpoint_dir", "/home/robert_tien/work/pachira/cnn-text-classification-tf/runs/1477945874/checkpoints/", "Checkpoint directory from training run")
# round 1
#tf.flags.DEFINE_string("checkpoint_dir", "/home/robert_tien/work/pachira/cnn-text-classification-tf/runs/1478039542/checkpoints/", "Checkpoint directory from training run")
# round 2
#tf.flags.DEFINE_string("checkpoint_dir", "/home/robert_tien/work/pachira/cnn-text-classification-tf/runs/1478052338/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_string("checkpoint_dir", OUT_DIR+sys.argv[3]+"/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
f.write("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
    f.write("{}={}\n".format(attr.upper(), value))
print("")

#pdb.set_trace()
# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test, _ = data_helpers.load_data_and_labels()
    y_test = np.argmax(y_test, axis=1)
elif len(sys.argv) > 1 and sys.argv[1]=="bootstrap":
    x_text, y_test, fnlist = data_helpers.load_data_and_labels_bootstrap_eval()
else: 
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
if sys.argv[1]=="bootstrap":
    fn = PATH+"chinesetokenization/msr_mecab_test/wiki.zh.text.vector"
    dictionary = readwordvec.build_wordvec_dict(fn)
    res = readwordvec.fit_transform(x_text, dictionary)
    x_test = np.array(res)
else: 
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")
f.write("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print("loading latest model from "+FLAGS.checkpoint_dir+"\n")
f.write("loading latest model from "+FLAGS.checkpoint_dir+"\n")
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("output/scores").outputs[0]
        #softmax= graph.get_operation_by_name("softmax_layer/softmax").outputs[0]
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_scores = np.empty(shape=[0, 2])

        for x_test_batch in batches:
            batch_scores, batch_predictions = sess.run([scores, predictions], {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            all_scores = np.concatenate([all_scores, batch_scores])

    softmax = tf.nn.softmax(all_scores)
    all_softmax = sess.run(softmax)
    print "=fjdsl=: files with prob. > 0.9 being the match"
    f.write("=fjdsl=: files with prob. > 0.9 being the match")
    cnt = 0
    #pdb.set_trace()
    l= np.where(all_softmax.transpose()[1] > 0.9)
    ll = l[0]
    ff = open(OUT_DIR+sys.argv[3]+"/class/"+sys.argv[4],"w")
    ff.write("mv=../../../../util/cpmv.sh \n")
    f.write("mv=../../../../util/cpmv.sh \n")
    ff.write("############greater than 0.9##################\n")
    for x in np.where(all_softmax.transpose()[1] > 0.9)[0]:
        print str(x)+" "+str(all_softmax.transpose()[1][x])
        print "$mv "+fnlist[x]+sys.argv[5]
        f.write(str(x)+" "+str(all_softmax.transpose()[1][x])+"\n")
        f.write("$mv "+fnlist[x]+"\t"+sys.argv[5]+"\n")
        ff.write("$mv "+fnlist[x]+"\t"+sys.argv[5]+"\n")
        cnt += 1
    print "total number of samples selected="+str(cnt)
    f.write("total number of samples selected="+str(cnt)+"\n")
    ff.write("############greater than 0.8 ####################\n")
    cnt = 0
    for x in np.where(all_softmax.transpose()[1] > 0.8)[0]:
        print str(x)+" "+str(all_softmax.transpose()[1][x])
        print "$mv "+fnlist[x]+sys.argv[5]
        f.write(str(x)+" "+str(all_softmax.transpose()[1][x])+"\n")
        f.write("$mv "+fnlist[x]+"\t"+sys.argv[5]+"\n")
        ff.write("$mv "+fnlist[x]+"\t"+sys.argv[5]+"\n")
        cnt += 1
    print "total number of samples selected="+str(cnt)
    f.write("total number of samples selected="+str(cnt)+"\n")
    ff.write("############greater than 0.7 ####################\n")
    cnt = 0
    for x in np.where(all_softmax.transpose()[1] > 0.7)[0]:
        print str(x)+" "+str(all_softmax.transpose()[1][x])
        print "$mv "+fnlist[x]+sys.argv[5]
        f.write(str(x)+" "+str(all_softmax.transpose()[1][x])+"\n")
        f.write("$mv "+fnlist[x]+"\t"+sys.argv[5]+"\n")
        ff.write("$mv "+fnlist[x]+"\t"+sys.argv[5]+"\n")
        cnt += 1
    print "total number of samples selected="+str(cnt)
    f.write("total number of samples selected="+str(cnt)+"\n")


# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    f.write("Total number of test examples: {}\n".format(len(y_test)))
    f.write("Accuracy: {:g}\n".format(correct_predictions/float(len(y_test))))

if len(sys.argv) > 2:
    f.close()
    ff.close()
