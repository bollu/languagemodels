#!/usr/bin/env python2
import re
from collections import *
import torch
import torch.nn as nn
import torch.optim as optim
import fire
# from prompt_toolkit import prompt


torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#                                   o1  o2         on
#                                G2O^   ^          ^
#h0 -> h1 H2H> h2 > h3 ... hn H2G> g1  G> g2 > ... > gn
#    I2H^   ^    ^      ^
#      i1  i2  i3  ... in
class RNN:
    def __init__(self, HSIZE, EMBEDSIZE, GSIZE):
        # hidden stat
        # 1 x HSIZE
        self.H0 = torch.randn((1, HSIZE), requires_grad=True,
                              dtype=torch.float32, device=device)
        self.I2H = torch.randn((EMBEDSIZE, HSIZE), requires_grad=True,
                               dtype=torch.float32, device=device)
        self.H2H = torch.randn((HSIZE, HSIZE), requires_grad=True,
                               dtype=torch.float32, device=device)
        self.H2G = torch.randn((HSIZE, GSIZE), requires_grad=True,
                               dtype=torch.float32,device=device)
        self.G2G = torch.randn((GSIZE, GSIZE), requires_grad=True,
                               dtype=torch.float32, device=device)
        self.G2O = torch.randn((GSIZE, EMBEDSIZE), requires_grad=True,
                               dtype=torch.float32,device=device)
        self.H2HBIAS = torch.randn((1, HSIZE), requires_grad=True,
                                   dtype=torch.float32,device=device)
        self.G2GBIAS = torch.randn((1, GSIZE), requires_grad=True,
                                   dtype=torch.float32,device=device)
        self.G2OBIAS = torch.randn((1, EMBEDSIZE), requires_grad=True,
                                   dtype=torch.float32,device=device)

    def get_params(self):
        return [self.H0, self.I2H, self.H2H, self.H2G, 
                self.G2G, self.G2O, self.H2HBIAS, self.G2GBIAS, self.G2OBIAS]

    # predict next words
    # inputs: SENTENCELEN x EMBEDSIZE
    # output: EMBEDSIZE
    def fwd(self, inputs, sentencelen):
        h = self.H0
        for i in range(sentencelen):
            # update hidden state for each word in sentence
            h = torch.tanh(torch.matmul(h, self.H2H) +  \
                           torch.matmul(inputs[i], self.I2H) +  \
                           self.H2HBIAS)

        g = torch.matmul(h, self.H2G)
        return torch.tanh(torch.matmul(g, self.G2O) + self.G2OBIAS)

    def save_dict(self):
        return {"H0": self.H0, 
                "I2H": self.I2H,
                "H2H":self.H2H, 
                "H2G": self.H2G,
                "G2G": self.G2G,
                "G2O": self.G2O,
                "H2HBIAS": self.H2HBIAS, 
                "G2GBIAS": self.G2GBIAS,
                "G2OBIAS": self.G2OBIAS
                }
    def load_from_dict(self, load):
        self.H0 = load["H0"]
        self.I2H = load["I2H"]
        self.H2H = load["H2H"]
        self.H2G = load["H2G"]
        self.G2G = load["G2G"]
        self.G2O = load["G2O"]
        self.H2HBIAS = load["H2HBIAS"]
        self.G2GBIAS = load["G2GBIAS"]

        self.H0.to(device)
        self.I2H.to(device)
        self.H2H.to(device)
        self.H2G.to(device)
        self.G2G.to(device)
        self.G2O.to(device)
        self.H2HBIAS.to(device)
        self.G2GBIAS.to(device)

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

def load_quick_brown_fox():
    with open("./corpora/quick-brown-fox.txt", "r") as f:
        sentences = []
        for l in f.read().split("."):
            l = l.replace("\n", " ")
            l = l.strip()
            sentences.append([w for w in l.split() if w])
    vocab = Counter([w for s in sentences for w in s])
    return (sentences, vocab)

def calc_num_params(params):
    sz = torch.tensor(1)
    for t in params:
        sz += torch.prod(torch.tensor(t.size()))
    return sz.item()

# return sliding windows of length l, followed by predict window of length m
def windows(l, xs):
    for i in range(len(xs) - l):
        yield (xs[i:i+l], xs[i+l])

# number of windows returned from windows
def nwindows(l, xs):
    return max(0, len(xs) - l)

# V: EMBEDSIZE, W: EMBEDSIZE
def cosine(v, w):
    return torch.dot(v.view(-1), w.view(-1)) / v.norm() / w.norm()

# return index in embeds of vector closest to v
# embeds: EMBEDSIZE
def get_closest_vector_ix(embeds, v):
    bestix = 0
    bestd = 1000
    with torch.no_grad():
        for i in range(embeds.size()[0]):
            d = torch.dot(embeds[i], v.reshape(-1))
            if d < bestd:
                bestix = i
                bestd = d
    return bestix




def train():
    SENTENCELEN = 2
    NEPOCHS = 3000
    EMBEDSIZE = 5
    HSIZE = 5
    GSIZE = 5

    ss, vcount = load_quick_brown_fox()
    vocab = set(vcount)
    vocab2ix = dict(zip(vocab, range(len(vocab))))
    ix2vocab = dict([(ix, w) for (w, ix) in vocab2ix.items()])
    vocabsize = len(vocab)
    # embeddings, jointly trained with the model
    embeds = torch.randn((vocabsize, EMBEDSIZE), device=device, requires_grad=True)

    # encode the ss into the corpus
    ss = [[vocab2ix[w] for w in s] for s in ss]

    model = RNN(HSIZE, EMBEDSIZE, GSIZE)
    optimizer = optim.SGD([embeds] + model.get_params(), lr=0.01)
    print("number of hyperparameters of model: %s" % 
            calc_num_params(model.get_params()))

    totalsize = 0
    for s in ss:
        totalsize += nwindows(SENTENCELEN, s)
    totalsize *= NEPOCHS


    # for each sentence, take window of word size and then
    iteration = 0

    for _ in range(NEPOCHS):
        for s in ss:
            for (in_, out_) in windows(SENTENCELEN, s):
                iteration += 1
                optimizer.zero_grad()
                # EMBEDSIZE
                encin_ = embeds[in_]
                # EMBEDSIZE
                encout_ = embeds[out_]
                # EMBEDSIZE
                predict = model.fwd(encin_, SENTENCELEN)

                # loss is how far apart they are in cosine similarity
                # loss = 1
                loss = cosine(encout_, predict)
                loss.backward()
                optimizer.step()

                if iteration % 1000 >= 998:
                    decodepredict = ix2vocab[get_closest_vector_ix(embeds, predict)]
                    print("%4.2f |  loss: %4.2f" % ((100.0 * iteration) / totalsize,  loss, ))
                    print("\t%s (%s | %s) " % ([ix2vocab[i] for i in in_], ix2vocab[out_], decodepredict))

    savedict = model.save_dict()
    savedict.update({"embeds": embeds, 
        "vocab2ix": vocab2ix, 
        "hsize":HSIZE,
        "gsize":GSIZE,
        "embedsize":EMBEDSIZE,
        "sentencelen":SENTENCELEN})
    print("H2Hbias: %s" % (savedict["H2HBIAS"], ))
    torch.save(savedict, "model.pth")


# returns [(word, dot product of v with embed[word])]
def closestWordsDesc(ix2vocab, embeds, v):
    dots = []
    for i in range(embeds.size()[0]):
        dots.append((ix2vocab[i], cosine(v, embeds[i])))
    # sort by cosine similarity (ascending)
    dots.sort(key=lambda x: x[1])
    # reverse for descending order
    dots.reverse()
    return dots


def repl():
    loaddict = torch.load("model.pth")
    embeds = loaddict["embeds"]
    vocab2ix = loaddict["vocab2ix"]
    EMBEDSIZE = loaddict["embedsize"] 
    HSIZE = loaddict["hsize"]
    GSIZE = loaddict["gsize"]
    ix2vocab = dict([(ix, w) for (w, ix) in vocab2ix.items()])

    model = RNN(HSIZE, EMBEDSIZE, GSIZE)
    model.load_from_dict(loaddict)

    while True:
        response = str(raw_input(">"))
        if not response: continue
        response = response.split(" ")
        if response[0] == "~" and len(response) == 2:
            w = response[1]
            if w not in vocab2ix: pass

            for (w, dist) in closestWordsDesc(ix2vocab, embeds, embeds[vocab2ix[w]]):
                print ("%20s %4.2f " % (w, dist))
        elif response[0] == "!" and len(response) >= 2:
            # 1(batchsize) x SENTENCELEN  X EMBEDSIZE
            encin_ = embeds[[vocab2ix[w] for w in response[1:]]]
            predict = model.fwd(encin_, len(response) - 1)
            decodepredict = ix2vocab[get_closest_vector_ix(embeds, predict)]
            print(decodepredict)



class CLI:
    def train(self):
        train()
    def repl(self):
        repl()


if __name__ == "__main__":
    fire.Fire(CLI)
    


