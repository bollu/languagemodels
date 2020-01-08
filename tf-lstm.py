#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
import numpy as np

import random
INPUTPATH='corpora/bible-kjv.txt'
EMBEDSIZE = 20
LEARNING_RATE=1e-3
NEPOCHS=1

with open(INPUTPATH, "r") as f:
  corpus = f.read()
  corpus = [w for w in corpus.split() if w]
  CORPUSLEN = CORPUSLEN
  vocab = set(corpus)
  
  # map words to their index in the embedding array
  VOCAB2IX = {w: i for (i, w) in enumerate(vocab)}
  VOCABSIZE = len(vocab)

  corpusix = np.empty(CORPUSLEN, dtype=np.int32)
  for i in range(CORPUSLEN):
      corpusix[i] = VOCAB2IX[corpus[i]]

assert VOCABSIZE is not None
assert CORPUSLEN is not None


def make_attention(s, e, name):
  """
  s: sentence: slen x ssz
  e: ssz x embedsz
  a: attention coefficients. Here is a design choice.
     either I have:
     - out[w][k] = Σw' s[w][i] a[i][j][k] s[w'][k] | a: embedsz^3
     - out[w][k] = Σw' a[i][k] (s[w];s[w'])[i] | a: 2xembedsz^2
     - out[w][k] = a[0][i][k]s[w][i] +  Σw' a[1][i][k]s[w'][i] | a: 2xembedsz^2
  """

  embedsz = e.shape[-1]

  a = tf.Variable(tf.random_normal([2*embedsz, embedsz]), name=name)

  # s: slen x ssz @ ssz x embedsz: slen x embedsz 
  s = tf.matmul(s, e)

  # ss: slen x (embedz + embedz)
  ss = tf.concat([s, s], axis=1)

  sattn = tf.matmul(ss, a)

  # softmax so we have keys into the next embedding layer
  # make sure that we know what values to use
  sattn = tf.softmax(sattn / tf.sqrt(embedsz))
  return a, sattn


var_syn0 = tf.Variable(tf.random_normal([VOCABSIZE, EMBEDSIZE]), name="syn0")
var_syn1neg = tf.Variable(tf.random_normal([VOCABSIZE, EMBEDSIZE]), name="syn1neg")

ph_fix = tf.placeholder(tf.int32, name="ph_fix")
ph_cix = tf.placeholder(tf.int32, name="ph_cix")
ph_label = tf.placeholder(tf.float32, name="ph_label")

# loss = (label - (focus[fix] . ctx[cix])^2
var_d = tf.tensordot(var_syn0[ph_fix, :], var_syn1neg[ph_cix, :], 1)
# loss = tf.norm(tf.math.sub(ph_label, d), name="loss")
var_loss = tf.norm(ph_label - var_d, name="loss")

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(var_loss)

# Step 1: _build the program_ you want to run
# Step 2: ask TF To kindly compile this program
# Step 3: push data through this program


def epoch(curepoch, sess):
  i = 0
  r = np.uint32(1)
  for ixf in range(CORPUSLEN):
    l = max(0, ixf - WINDOWSIZE)
    r = min(CORPUSLEN - 1, ixf + WINDOWSIZE)
  
    # word(fox) -> index (10) -> vector [1; -1; 42; -42]
    data_fix = corpusix[ixf] # VOCAB2IX[corpus[ixf]]
  
    
    # the fox [vc=jumps *vf=over* the] dog (vc.vf=1)
    for ixc in range(l, r):
      # variable[placeholder]
      data_cix = corpusix[ixc] # VOCAB2IX[corpus[ixc]]
      data_label = 1
  
    # vc=the fox [jumps *vf=over* the] dog (vc.vf = 0)
    for _ in range(NEGSAMPLES):
      r =  r * 25214903917 + 11
      data_cix = r % (VOCABSIZE - 1)
      data_label = 0
    
    # print("fix: %s | cix: %s | label: %s" % (data_fix, data_cix, data_label))
    DEBUGGING_DATA_BOTTLENECK = False
    if DEBUGGING_DATA_BOTTLENECK:
        loss = 0
    else:
     loss, _ = sess.run([var_loss, optimizer], 
 	    feed_dict={ph_fix:data_fix, ph_cix: data_cix, ph_label: data_label})

    if i % 4 == 0: print("loss: %10.2f | %5.2f%%" % (loss, (curepoch + (ixf / CORPUSLEN)) / NEPOCHS * 100.0))
    i += 1

with tf.Session() as sess:
  global_init = tf.global_variables_initializer()
  sess.run(global_init)
  for i in range(NEPOCHS):
    print("===epoch: %s===" % i)
    epoch(i, sess) 

  data_syn0 = sess.run([var_syn0])
  data_syn1neg = sess.run([var_syn1neg])

  # TASK 1. Pull out the data from the session, and _print it_. Maybe try and
  # implement distance()
  
  # print distance of fox from all other words, ordered by ascending order (Dot
  # product / cosine distance)
  distance('fox', data_syn0)

  # quick - fox + dog == ? print best candidates for this
  # Fox :  quick :: fox : ? == (quick - fox) + fox = quick
  analogy('fox', 'quick', 'dog', data_syn0)
  
  # TASK 2. copy and understand (plz plz plz) the data saving/loading code, and
  # save the learnt word vectors.

  # TASK 3. make this batced: use multiple indeces and
  # multipl labels _in batch mode_. I presume this is requires one to
  # change the code to "store" the (fix, cix, and labels) and then
  # pass them as arrays to sess.run(...)
