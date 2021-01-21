from gensim.models.word2vec import Text8Corpus
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import PathLineSentences
import logging
import re


# filname = 'model_1bshort_m_1_1_1_1_t_10_1_1_1_trigram_model.pkl'
filname = 'model_1bshort_h1000_m_1_1_1_1_t_10_10_1_1_trigram_model.pkl'
log_filname = re.sub('model', 'gen_log', re.sub('.pkl', '.txt', filname))
corpus = '1bshort.txt'
# corpus = '1bshort_h1000.txt'
# corpus = '1bshort_h1000 - Copy.txt'

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S',
                    level=logging.INFO,
                    filename='result/' + log_filname)

sentences = PathLineSentences(corpus)


ngram_phrases = Phrases.load(
    "./result/" + filname)
score_ngram = sorted(list(set(ngram_phrases.export_phrases(
    sentences))), key=lambda x: x[1], reverse=True)


ngram = Phraser(ngram_phrases)
print(score_ngram[:10])

# ----------------------------Outputs
contextes = []
for ngram_byte in score_ngram[:10]:
    ngram = ngram_byte[0].decode("utf-8")
    contexte = []
    for sentence in sentences:
        if ngram in ' '.join(sentence):
            contexte.append(' '.join(sentence))
        if len(contexte) >= 1:
            break
    contextes.append(contexte[0])
    print(ngram)
    print(contexte)
print(contextes)
