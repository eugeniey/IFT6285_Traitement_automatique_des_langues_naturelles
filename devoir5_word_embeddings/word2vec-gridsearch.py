#!/usr/bin/env python3
# felipe@ift6285 aout 2019

"""
entrainer des embeddings avec gensim
"""

import multiprocessing
from time import time
import argparse
import logging  # Setting up the loggings to monitor gensim
import sys

import gensim
from gensim.test.utils import datapath
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import PathLineSentences

from gensim.models.callbacks import CallbackAny2Vec

# ------------------------------------------------------------------
# ------------------------------------------------------------------

prog = 'word2vec-gensim-train'

# ---------------------------------------------
#        gestion ligne de commande
# ---------------------------------------------

def get_args():

    parser = argparse.ArgumentParser(
        description="train a gensim word2vec model from text read on stdin")

    parser.add_argument("-v", '--verbosity', type=int,
                        help="increase output verbosity", default=0)
    parser.add_argument("-n", '--nb', type=int,
                        help="# of input lines to read (per file I think)", default=None)
    parser.add_argument("-d", '--datapath', type=str,
                        help="directory where txt files can be found", default=None)
    parser.add_argument("-f", '--name', type=str,
                        help="basename modelname", default="genw2v")
    parser.add_argument("-s", '--size', type=int,
                        help="dim of vectors", default=300)
    parser.add_argument("-g", '--negative', type=int,
                        help="# neg samples", default=100)
    parser.add_argument("-c", '--mincount', type=int,
                        help="min count of a word", default=100)
    parser.add_argument("-w", '--window', type=int,
                        help="window size", default=2)
    parser.add_argument("-a", '--alpha', type=float,
                        help="alpha", default=0.03)
    parser.add_argument("-m", '--minalpha', type=float,
                        help="min alpha", default=0.0007)
    parser.add_argument("-e", '--epochs', type=int,
                        help="epochs", default=10)
    parser.add_argument("-l", '--loss', action="store_true")
    parser.add_argument('outdir', type=str,
                        help="path to output models' files")

    return parser.parse_args()


size = [100, 150]            # size of embedding
min_count = [5, 10, 30]      # take words there min_count
window = [3, 5, 9]

sg = [0, 1]                   # training algo: 1 skipgram 0 CBOW
cbow_mean = [0, 1]           # only for CBOW when sg = 1
# hierarchical softmax will be used for model training or Negative sampling
hs = [0, 1]
negative = [0, 5, 10]        # how many “noise words” should be drawn
ns_exponent = [0, 0.75, 1]   # only use it with negative sampling >0

alpha = [0.001, 0.025, 0.05]
min_alpha = [0.001, 0.0001]
iterr = [5, 10, 30]
# The threshold for configuring which higher-frequency words are randomly downsampled,
sample = [0.01, 0.0001]
trim_rule = None            # min_count will be used
# ---------------------------------------------
#        tools
# ---------------------------------------------


def to_min(t):
    return round((time() - t) / 60, 2)


# init callback class
class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """

    def __init__(self):
        self.epoch = 0
        if args.nb is not None:
            self.nb = args.nb

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if args.nb is not None:
            if self.epoch == 0:
                logging.info('v_size:{}  epoch:{}  Loss:{}'.format(
                    self.nb, self.epoch, loss))
            else:
                logging.info('v_size:{}  epoch:{}  Loss:{}'.format(
                    self.nb, self.epoch, loss - self.loss_previous_step))
            self.epoch += 1
            self.loss_previous_step = loss

# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------


print(get_args())
args = get_args()

ext = "{}-size{}-window{}-neg{}-mincount{}-alpha{}-minalpha{}-epochs{}".format(args.name,
                                                                               args.size, args.window, args.negative, args.mincount, args.alpha, args.minalpha, args.epochs)

logname = f"{args.outdir}/{ext}.log"
modelname = f"{args.outdir}/{ext}.w2v"
txt_modelname = f"{args.outdir}/{ext}.txt"

logging.basicConfig(  # filename=logname,
    format="%(levelname)s %(asctime)s [" + \
    prog + "] %(message)s",
    datefmt='%H:%M:%S', level=logging.INFO,
    handlers=[
        logging.FileHandler(logname),
        logging.StreamHandler()
    ])

logging.info(
    f"running with config: {ext} -- hopefully generating {modelname} {txt_modelname}")

# 1) reading input into a list of words
t = time()
sents = []

from string import punctuation
import re
punct = set(punctuation)
punct_cleaned = set(punctuation.replace(
    "-", "").replace("'", "").replace(",", "").replace("(", " ").replace(")", " "))

if args.datapath is None:
    ntok = 0
    for line in sys.stdin:
        line = re.sub(' +', ' ', line)
        words = line.rstrip().split()
        # print(words)
        words = [''.join(ch for ch in word if ch not in punct_cleaned).lower()
                 for word in words if word not in punct_cleaned]
        words = [word for word in words if word not in ['']]
        # print(words)
        ntok += len(words)
        sents.append(words)
        # logging.info(f"Read {len(sents)} sentences, {ntok} tokens in {to_min(t)} minutes")
else:
    sents = PathLineSentences(args.datapath, limit=args.nb)


# 2) run the phraser
t = time()
phrases = Phrases(sents, min_count=10, progress_per=100000)
bigram = Phraser(phrases)

sents_b = bigram[sents]
logging.info(
    f"Phrases found  {len(phrases.vocab)} phrases in {to_min(t)} minutes")

if args.verbosity > 2:
    for i in range(0, 10):
        logging.debug("sent: ", sents_b[i])

# 3 let's train a model
cores = multiprocessing.cpu_count()  # Count the number of cores

for sg_ in sg:
    for hs_ in hs:
        for negativ_ in negative:
            for ns_exp_ in ns_exponent:
                for cbow_m in cbow_mean:
                    if sg_ == 1 and cbow_m == 1:
                        continue

                    save_ext = "{}-size{}-window{}-neg{}-mincount{}-alpha{}-minalpha{}-epochs{}-sg_{}-hs_{}-negativ_{}-ns_exp_{}-cbow_m{}".format(args.name,
                                                                                                                                         args.size, args.window, args.negative, args.mincount, args.alpha, args.minalpha, args.epochs, sg_, hs_, negativ_, ns_exp_,
                                                                                                                                         cbow_m)

                    savemodelname = f"{args.outdir}/{save_ext}.w2v"
                    txt_savemodelname = f"{args.outdir}/{save_ext}.txt"

                    w2v_model = Word2Vec(min_count=args.mincount,
                                         window=args.window,
                                         size=args.size,
                                         sample=6e-5,
                                         alpha=args.alpha,
                                         min_alpha=args.minalpha,
                                         iter=args.epochs,
                                         workers=cores - 1,
                                         sg=sg_, hs=hs_, negative=negativ_, ns_exponent=ns_exp_,
                                         cbow_mean=cbow_m)

                    t = time()
                    w2v_model.build_vocab(sents_b, progress_per=10000)
                    logging.info(
                        'Time to build vocab: {} mins'.format(to_min(t)))

                    t = time()
                    w2v_model.train(sents_b, total_examples=w2v_model.corpus_count,
                                    epochs=args.epochs, report_delay=1)
                    logging.info(f"Time to train the model: {to_min(t)} mins")

                    if args.nb is not None:
                        logging.info(f"v_size: {args.nb}  time: {to_min(t)}")

                    # stream the model
                    w2v_model.init_sims(replace=True)

                    # save it
                    # I must read the doc, but as far as I understand, the save function is good if the model
                    # is read with gensim
                    #
                    w2v_model.save(savemodelname)
                    logging.info(f"saved {savemodelname}")

                    # this one exports a textfile that spacy can (hopefully) convert
                    w2v_model.wv.save_word2vec_format(txt_savemodelname)
                    logging.info(f"saved {txt_savemodelname}")

                    # 4) be gentle
                    logging.info(
                        f"ending -- log: {logname} -- hopefully generated {savemodelname}* and {txt_savemodelname}")
