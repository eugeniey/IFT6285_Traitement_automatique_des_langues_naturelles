#!/usr/bin/python3


# pip install editdistance==0.3.1
# pip install pyxDamerauLevenshtein
# pip install nwalign
# pip install libindic-soundex
# pip install libindic-utils
# Python corrige.py voc-1bwc.txt --train devoir3-train.txt --test devoir3-train.txt > sortie.txt
# Python corrige.py voc-1bwc.txt --test devoir3-train.txt


import re
import sys
import argparse
from collections import Counter
import time
import heapq
from string import punctuation
import editdistance as Levenshtein
from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance
from libindic.soundex import Soundex
import jellyfish


# ----------------------------Inputs

# use stdin if it's full
input_stream = None
if not sys.stdin.isatty():
    input_stream = sys.stdin
    data = input_stream.readlines()
    data = [line.rstrip() for line in data]

# read option usin argparser
argparser = argparse.ArgumentParser(description="Option d'entrainement")
argparser.add_argument('vocab', type=str, nargs=1)
argparser.add_argument('--train', type=str, default=None,
                       help='donne d\'entrainement')
argparser.add_argument('--test', type=str, default=None,
                       help='donne de test')
argparser.add_argument('--distance', type=str, default='lv',
                       help='type de distance. lv : Levenshtein distance; dlv:Damerau-Levenshtein; sd:soundex')
argparser.add_argument('--mode', type=int, default=3,
                       help='1:distance 2:vraisemblance    3:les deux')
argparser.add_argument('-v', '--verbose', action="store_false",
                       help="verbose output")
args = argparser.parse_args()


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


vocab = {}
punct = set(punctuation.replace("-", "").replace("'", ""))

with open(args.vocab[0], 'r', encoding="utf8") as f:
    for line in f:
        s = ''.join(ch for ch in line if ch not in punct)
        (val, key) = re.sub('\n', '', line.lstrip()).split(' ', maxsplit=1)
        if(hasNumbers(key)):
            continue
        vocab[key] = int(val)

if args.train:
    train = []
    with open(args.train, 'r', encoding="utf8") as f:
        for i, line in enumerate(f):
            (right, wrong) = re.sub('\n', '', line.lstrip()).split('\t', maxsplit=1)
            train.append((right, wrong))

if args.test:
    test = []
    with open(args.test, 'r', encoding="utf8") as f:
        for i, line in enumerate(f):
            (right, wrong) = re.sub('\n', '', line.lstrip()).split('\t', maxsplit=1)
            test.append((right, wrong))

if args.distance:
    pass
if args.mode:
    pass
if args.verbose:
    pass
#---------------------------variables globales
NB_CORR = 5
INSTANCE = Soundex()
WORDS = Counter(vocab)
WORDS_SET = set(w for w in WORDS)
args.distance = "dlv"


WORDS_SOUND = {}
for word, value in WORDS.items():
    code = INSTANCE.soundex(word)
    WORDS_SOUND.setdefault(code, []).append(word)


if input_stream:
    INPUT_DATA = data
if args.train:
    TRAIN_DATA = [x for x, y in train][0:100]
if args.test:
    TEST_DATA = test
#-------------------------- Spelling Corrector


def measure_distance(word1, word2, distance_type):
    if distance_type == 'lv':
        distance = Levenshtein.eval(word1, word2)
    if distance_type == 'dlv':
        distance = jellyfish.damerau_levenshtein_distance(word1, word2)
    if distance_type == 'jw':
        # Jaro–Winkler indicates the similiraty, we take the inverse
        distance = -jellyfish.jaro_winkler_similarity(word1, word2)
    if distance_type == 'j':
        distance = -jellyfish.jaro_similarity(word1, word2)
    if distance_type == 'hm':
        distance = jellyfish.hamming_distance(word1, word2)
    return distance


def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N
    # return - WORDS.get(word, 0)


def score(word, wrong_word, distance_type, mode):
    if mode == 1:
        score = -P(word)
    if mode == 2:
        score = measure_distance(word, wrong_word, distance_type)
    if mode == 3:
        score = [measure_distance(word, wrong_word, distance_type), -P(word)]
    return score


def top_correction(word, N=1):
    "Most probable spelling correction for word."
    # return sorted(candidates(word), key=lambda x: score(x, word, args.distance, args.mode), reverse=False)[:N]
    return heapq.nsmallest(N, candidates(word), key=lambda x: score(x, word, args.distance, args.mode))


def candidates(word):
    "Generate possible spelling corrections for word."
    sound_code = INSTANCE.soundex(word)
    candidates = set(WORDS_SOUND[sound_code] if sound_code in WORDS_SOUND else
                     [word])
    candidates = WORDS_SET
    return candidates


def spelltest(tests, verbose=False):
    "Run correction(wrong) on all (right, wrong) pairs; report results."
    start = time.time()
    good, unknown = 0, 0
    n = len(tests)
    for wrong, right in tests:
        w = top_correction(wrong, 1)[0]
        good += (w == right)
        if w != right:
            unknown += (right not in WORDS)
            if verbose:
                print('correction({}) => {} ({}); expected {} ({})'
                      .format(wrong, w, WORDS[w], right, WORDS[right]))
    dt = time.time() - start
    print('{:.0%} of {} correct ({:.0%} unknown) at {:.0f} words per second '
          .format(good / n, n, unknown / n, n / dt))


def spelltest_first5(tests, verbose=False):
    "Run correction(wrong) on all (right, wrong) pairs; report results."
    start = time.time()
    good, unknown = 0, 0
    n = len(tests)
    for wrong, right in tests:
        w = top_correction(wrong, 5)

        if(w[0] == right):
            good += 1
        # We look into the other 5 correction
        else:
            for correction in w[1:]:
                if(correction == right):
                    good += 0.5

        if w != right:
            unknown += (right not in WORDS)
            if verbose:
                print('correction({}) => {} ({}); expected {} ({})'
                      .format(wrong, w[0], WORDS[w[0]], right, WORDS[right]))
    dt = time.time() - start
    print('{:.0%} of {} correct ({:.0%} unknown) at {:.0f} words per second '
          .format(good / n, n, unknown / n, n / dt))


def spellcheck(wrongs):
    "output au format en consigne dans le devoir"
    start = time.time()
    for wrong in wrongs:
        print(wrong, end='')
        for correction in top_correction(wrong, NB_CORR):
            print('\t', end='')
            print(correction, end='')
        print('\n',  end='')
    dt = time.time() - start
    print('{} words corrected in {:.0f} minutes'
          .format(len(wrongs), dt / 60))


if input_stream:
    spellcheck(INPUT_DATA)
if args.train:
    spellcheck(TRAIN_DATA)
if args.test:
    spelltest_first5(TEST_DATA, verbose=args.verbose)
