#!/usr/bin/python3
from gensim.models.word2vec import Text8Corpus
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import PathLineSentences
import logging
import argparse
import time


# python ngram_train.py --corpus 1bshort --order 5 --mincount 5 5 5 5 --threshold 0 0 0 0 --vocab_prop 10 --log --scoring npmi >> ./output_fivegram.txt

# ----------------------------Inputs
# read option usin argparser
argparser = argparse.ArgumentParser(description="Option d'entrainement")
argparser.add_argument('--corpus', type=str, default='1bshort_h1000',
                       help='corpus d\'entrainement')
argparser.add_argument('--order', type=str, default=3,
                       help='ordre des ngramme a entrainer')
argparser.add_argument('--mincount', nargs="+", type=int, default=[1, 1, 1, 1],
                       help=' Ignore all words and bigrams with total collected count lower than this value.')
argparser.add_argument('--threshold', nargs="+", type=int, default=[1, 1, 1, 1],
                       help='Represent a score threshold for forming the phrases')
argparser.add_argument('--scoring', type=str, default='default',
                       help='score method npmi')
argparser.add_argument('--vocab_prop', type=float, default=1.0,
                       help='portion of vocab to train')
argparser.add_argument('-l', '--log', action="store_true",
                       help="log time and other useful element")
argparser.add_argument('-s', '--save', action="store_true",
                       help="save model")
args = argparser.parse_args()

if args.corpus:
    filename = args.corpus
# Load training data.
# filename = '1bshort.txt'
    sentences = PathLineSentences(filename + '.txt')
if args.order:
    pass
if args.mincount:
    pass
if args.threshold:
    pass
if args.scoring:
    pass
if args.vocab_prop:
    sentences = list(sentences)
    sentences = sentences[0: int(len(sentences) * args.vocab_prop / 10)]


from string import punctuation
punct_cleaned = set(punctuation.replace("-", "").replace("'", ""))
sents = PathLineSentences(filename + '.txt')
sentences = list(sents)
sentences_clean = []
for sentence in sentences:
    sentence_new = []
    for word in sentence:
        if word not in punct_cleaned:
            word = ''.join(ch for ch in word if ch not in punct_cleaned)
            sentence_new.append(word.lower())
    sentences_clean.append(sentence_new)
sentences = sentences_clean
# ----------------------------logging
if args.log == True:
    threshold = '_'.join([str(integer) for integer in args.threshold])
    mincount = '_'.join([str(integer) for integer in args.mincount])
    log_filname = 'result/train_logs_' + filename + '_o_' + \
        str(args.order) + '_m_' + mincount + '_t_' + threshold + '.txt'
    logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S',
                        level=logging.INFO,
                        filename=log_filname)


def log_train(ngram, vocab_len, train_time, ngram_len, maximum, minimum, min_count, threshold, top_ngram):
    log_str = 'filename: ' + str(filename) + '\t' \
        + 'ngram: ' + str(ngram) + '\t' \
        + 'ngram_len: ' + str(ngram_len) + '\t' \
        + 'train_time: ' + str(train_time) + '\t' \
        + 'vocab_len: ' + str(vocab_len) + '\t' \
        + 'max: ' + str(maximum) + '\t' \
        + 'min: ' + str(minimum) + '\t' \
        + 'train_time: ' + str(train_time) + '\t' \
        + 'min_count: ' + str(min_count) + '\t' \
        + 'threshold: ' + str(threshold) + '\t' \
        + 'top_ngram: ' + str(top_ngram)
    print(log_str)
# ----------------------------Entrainement


# Train bigram model.
if (int(args.order) == 2 and args.save == True) or (int(args.order) >= 2 and args.save == False):
    start = time.time()
    bigram_phrases = Phrases(
        sentences, min_count=args.mincount[0], threshold=args.threshold[0], scoring='npmi')
    bigram = Phraser(bigram_phrases)
    if args.save == True:
        bigram_phrases.save('result/model_' + filename + '_m_' +
                            mincount + '_t_' + threshold + '_bigram_model.pkl')
    score_bigram = sorted(list(set(bigram_phrases.export_phrases(
        sentences))), key=lambda x: x[1], reverse=True)
    train_time = time.time() - start
    ngram = score_bigram
    log_train('bigram', len(sentences), train_time,
              len(ngram), ngram[0][1], ngram[-1][1], mincount, threshold, ngram[:10])

# Train trigram model.
if (int(args.order) == 3 and args.save == True) or (int(args.order) >= 3 and args.save == False):
    start = time.time()
    if args.save == True:
        bigram_phrases = Phrases.load(
            'result/model_' + filename + '_m_' +
            mincount + '_t_' + threshold + '_bigram_model.pkl')
        bigram = Phraser(bigram_phrases)

    trigram_phrases = Phrases(
        bigram[sentences], min_count=args.mincount[1], threshold=args.threshold[1], scoring='npmi')
    trigram = Phraser(trigram_phrases)
    if args.save == True:
        trigram_phrases.save('result/model_' + filename + '_m_' +
                             mincount + '_t_' + threshold + '_trigram_model.pkl')
    score_trigram = sorted(list(filter(lambda e: str(e[0]).count('_') == 1,
                                       list(set(trigram_phrases.export_phrases(bigram[sentences]))))),
                           key=lambda x: x[1], reverse=True)
    train_time = time.time() - start
    ngram = score_trigram

    log_train('trigram', len(sentences), train_time,
              len(ngram), ngram[0][1], ngram[-1][1], mincount, threshold, ngram[:10])

# Train 4gram model.
if (int(args.order) == 4 and args.save == True) or (int(args.order) >= 4 and args.save == False):
    start = time.time()
    if args.save == True:
        trigram_phrases = Phrases.load(
            'result/model_' + filename + '_m_' +
            mincount + '_t_' + threshold + '_trigram_model.pkl')
        trigram = Phraser(trigram_phrases)

    fourgram_phrases = Phrases(
        trigram[sentences], min_count=args.mincount[2], threshold=args.threshold[2], scoring='npmi')
    fourgram = Phraser(fourgram_phrases)

    if args.save == True:
        fourgram_phrases.save('result/model_' + filename + '_m_' +
                              mincount + '_t_' + threshold + '_fourgram_model.pkl')
    score_fourgram = sorted(list(filter(lambda e: str(e[0]).count('_') == 2,
                                        list(set(fourgram_phrases.export_phrases(trigram[sentences]))))),
                            key=lambda x: x[1], reverse=True)
    train_time = time.time() - start

    ngram = score_fourgram
    log_train('fourgram', len(sentences), train_time,
              len(ngram), ngram[0][1], ngram[-1][1], mincount, threshold, ngram[:10])

    # Train 5gram model.
if (int(args.order) == 5 and args.save == True) or (int(args.order) >= 5 and args.save == False):
    start = time.time()
    if args.save == True:
        fourgram_phrases = Phrases.load(
            'result/model_' + filename + '_m_' +
            mincount + '_t_' + threshold + '_fourgram_model.pkl')
        fourgram = Phraser(fourgram_phrases)

    fivegram_phrases = Phrases(
        fourgram[sentences], min_count=args.mincount[3], threshold=args.threshold[3], scoring='npmi')
    fivegram = Phraser(fivegram_phrases)

    if args.save == True:
        fivegram_phrases.save('result/model_' + filename + '_m_' +
                              mincount + '_t_' + threshold + '_fivegram_model.pkl')

    score_fivegram = sorted(list(filter(lambda e: str(e[0]).count('_') == 3,
                                        list(set(fivegram_phrases.export_phrases(fourgram[sentences]))))),
                            key=lambda x: x[1], reverse=True)
    train_time = time.time() - start

    ngram = score_fivegram
    log_train('fivergram', len(sentences), train_time,
              len(ngram), ngram[0][1], ngram[-1][1], mincount, threshold, ngram[:10])
