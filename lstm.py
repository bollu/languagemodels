#!/usr/bin/env python3
import re
from collections import *
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

EMBEDSIZE = 3
HSIZE = 5           
GSIZE = 7           

#                                   o1  o2         on
#                                G2O^   ^          ^
#h0 -> h1 H2H> h2 > h3 ... hn H2G> g1  G> g2 > ... > gn
#    I2H^   ^    ^      ^
#      i1  i2  i3  ... in
class RNN:
    def __init__(self):
        # hidden stat
        # 1 x HSIZE
        self.H0 = torch.randn((1, HSIZE), requires_grad=True, dtype=torch.float32)
        self.I2H = torch.randn((EMBEDSIZE, HSIZE), requires_grad=True, dtype=torch.float32)
        self.H2H = torch.randn((HSIZE, HSIZE), requires_grad=True, dtype=torch.float32)
        self.H2G = torch.randn((HSIZE, GSIZE), requires_grad=True, dtype=torch.float32)
        self.G2G = torch.randn((GSIZE, GSIZE), requires_grad=True, dtype=torch.float32)
        self.G2O = torch.randn((GSIZE, EMBEDSIZE), requires_grad=True, dtype=torch.float32)

    def get_params(self):
        return [self.H0, self.I2H, self.H2H, self.H2G, self.G2G, self.G2O]

    # predict next words
    # inputs: SENTENCELEN x EMBEDSIZE
    # output: NPREDICT x EMBEDSIZE
    def fwd(self, inputs, npredict):
        h = self.H0
        for i in range(inputs.size()[0]):
            # update hidde state
            h = torch.tanh(h @ self.H2H + inputs[i] @ self.I2H)

        g = h @ self.H2G
        outs = []
        for i in range(npredict):
            outs.append(g @ self.G2O)
            g = g @ self.G2G
        return torch.stack(outs)

# remove the "xx:yy"  from a verse "xx:yy ..." and return the "..."
def remove_verse_number(s): 
    out = re.split("[0-9]+:[0-9]+", s)
    if len(out) > 1: return out[1].strip()
    return out[0]

# returns list of list of words (list of sentences) and a Counter of
# vocabulary
def load_bible():
    with open("./corpora/bible-kjv.txt", "r") as f:
        sentences = []
        for l in f.read().split("."):
            l = remove_verse_number(l)
            l = l.replace("\n", " ")
            l = l.replace(";", " ")
            l = l.replace(":", " ")
            l = l.replace("(", " ")
            l = l.replace(")", " ")
            l = l.strip()
            sentences.append([w for w in l.split() if w])

        vocab = Counter([w for s in sentences for w in s])
        return sentences, vocab

def calc_num_params(params):
    sz = torch.tensor(1)
    for t in params:
        sz += torch.prod(torch.tensor(t.size()))
    return sz.item()

# return sliding windows of length l, followed by predict window of length m
# In [3]: list(windows(3, 2, [1, 2, 3, 4, 5, 6, 7]))
# Out[3]: [([1, 2, 3], [4, 5]), ([2, 3, 4], [5, 6]), ([3, 4, 5], [6, 7])]
def windows(l, m, xs):
    for i in range(len(xs) - l - m + 1):
        yield (xs[i:i+l], xs[i+l:i+l+m])

def get_embedding_matrix(embeds, w2ix, words):
    out = []
    for w in words:
        out.append(embeds[w2ix[w]])
    return torch.stack(out)

if __name__ == "__main__":
    SENTENCELEN = 5
    PREDICTLEN = 3

    ss, vcount = load_bible()
    vocab = set(vcount)
    vocab2ix = dict(zip(vocab, range(len(vocab))))
    vocabsize = len(vocab)
    # embeddings, jointly trained with the model
    embeds = torch.randn((vocabsize, EMBEDSIZE), requires_grad=True)

    model = RNN()
    optimizer = optim.SGD([embeds] + model.get_params(), lr=1e-2)
    print("number of hyperparameters of model: %s" % calc_num_params(model.get_params()))

    # for each sentence, take window of word size and then
    for s in ss:
        for (in_, out_) in windows(SENTENCELEN, PREDICTLEN, s):
            in_ = get_embedding_matrix(embeds, vocab2ix, in_)
            out_ = get_embedding_matrix(embeds, vocab2ix, out_)
            predict = model.fwd(in_, PREDICTLEN)

