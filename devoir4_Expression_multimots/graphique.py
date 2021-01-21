from gensim.models.phrases import Phrases, Phraser
import os
import io
from gensim.models.word2vec import Text8Corpus
from gensim.test.utils import datapath
from gensim.models.word2vec import PathLineSentences
import matplotlib.pyplot as plt
import pandas as pd
import time
import csv
import numpy as np

PATH_TRAINING = "C:/Users/Euge/Documents/Session1-Maitrise/training_1bwc/"

# ===============================================================
# Informations pour les bigrammes

sents = PathLineSentences(PATH_TRAINING,limit = 200000) 
list_sentences = list(sents)
len_total_sentences = len(list_sentences)

number_points = 15

sentences_taken = []
bigramme_taken  = []
time_taken      = []

for i in range(1,number_points+1):
    print(i)
    
    start = time.time()
    
    sentences = list_sentences[0: int(i * (len_total_sentences/number_points))]
    
    bigram_phrases = Phrases(sentences, min_count=1, threshold=10)
    score_bigram = sorted(list(set(bigram_phrases.export_phrases(
    sentences))), key=lambda x: x[1], reverse=True)

    end = time.time()
    time_taken.append(end-start)
    sentences_taken.append(len(sentences))
    bigramme_taken.append(len(score_bigram))

with open("10mil_text_sentences_bigramme_time.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(zip(sentences_taken, trigramme_taken, tri_time_taken))

# ===============================================================
# Informations pour les trigrammes

sents = PathLineSentences(PATH_TRAINING,limit = 160000) 
list_sentences = list(sents)
len_total_sentences = len(list_sentences)
number_points = 9


sentences_taken = []
trigramme_taken = []
tri_time_taken  = []

for i in range(1,number_points+1):
    print(i)
    
    start = time.time()
    
    sentences = list_sentences[0: int(i * (len_total_sentences/number_points))]
    
    bigram_transformer = Phrases(sentences, min_count=1, threshold=10, delimiter = b' ')
    bt = bigram_transformer[sentences]
    trigram_transformer = Phrases(bt, min_count=1, threshold=10, delimiter = b' ')
    
    score_trigram = sorted(list(filter(lambda e:str(e[0]).count(' ') == 2,
                             list(set(trigram_transformer.export_phrases(bt))))), 
                             key=lambda x: x[1], reverse=True )

    end = time.time()
    tri_time_taken.append(end-start)
    sentences_taken.append(len(sentences))
    trigramme_taken.append(len(score_trigram))

with open("10mil_text_sentences_trigramme_time_2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(zip(sentences_taken, trigramme_taken, tri_time_taken))


# ===============================================================
# Informations pour les 4-grammes


sents = PathLineSentences(PATH_TRAINING,limit = 120000)   #200000) 
list_sentences = list(sents)
len_total_sentences = len(list_sentences)
number_points = 9


sentences_taken = []
qrtgramme_taken = []
qrt_time_taken  = []

for i in range(1,number_points+1):
    print(i)
    
    start = time.time()
    
    sentences = list_sentences[0: int(i * (len_total_sentences/number_points))]
    
    bigram_transformer = Phrases(sentences, min_count=1, threshold=10, delimiter = b' ')
    bt = bigram_transformer[sentences]

    trigram_transformer = Phrases(bt, min_count=1, threshold=10, delimiter = b' ')
    tt = trigram_transformer[sentences]

    qrtgram_transformer = Phrases(tt, min_count=1, threshold=10, delimiter = b' ')
    
    score_4gram = sorted(list(filter(lambda e:str(e[0]).count(' ') == 3,
                             list(set(qrtgram_transformer.export_phrases(tt))))), 
                             key=lambda x: x[1], reverse=True )

    end = time.time()
    qrt_time_taken.append(end-start)
    sentences_taken.append(len(sentences))
    qrtgramme_taken.append(len(score_4gram))


with open("10mil_text_sentences_4gramme_time_2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(zip(sentences_taken, qrtgramme_taken, qrt_time_taken))



# ===============================================================
# Création des graphiques


file_tri = pd.read_csv("10mil_text_sentences_trigramme_time_2.csv", header=None).dropna()
file_bi = pd.read_csv("10mil_text_sentences_bigramme_time.csv", header=None).dropna()
file_4 = pd.read_csv("10mil_text_sentences_4gramme_time.csv", header=None).dropna()

sentences_taken_4 = file_4.iloc[:,0].astype(float)
qrtgramme_taken = file_4.iloc[:,1].astype(float)
time_4 = file_4.iloc[:,2].astype(float)

sentences_taken_tri = file_tri.iloc[:,0].astype(float)
trigramme_taken = file_tri.iloc[:,1].astype(float)
time_tri = file_tri.iloc[:,2].astype(float)

sentences_taken_bi = file_bi.iloc[:,0].astype(float)
bigramme_taken = file_bi.iloc[:,1].astype(float)
time_bi = file_bi.iloc[:,2].astype(float)


plt.plot(sentences_taken_bi[0:-3], bigramme_taken[0:-3], '-', color = '#FF0000', label = "Bigrammes")
plt.plot(sentences_taken_tri, trigramme_taken, '-', color = '#1e4ff1', label = "Trigrammes")
plt.plot(sentences_taken_4, qrtgramme_taken, '-', color = '#32CD32', label = "4-grammes")
#plt.xticks([16, 30, 50, 100, 200, 220, 240, 260, 280,  300, 400, 500,600] )
plt.xticks(fontsize=9)
plt.xticks(np.arange(0e6, 9e6, 1e6))
plt.ticklabel_format(style='sci', axis='x', scilimits=(6,6))
plt.ticklabel_format(style='sci', axis='y', scilimits=(5,5))
plt.xlabel('Nombre de phrases', fontsize = 10)
plt.ylabel('Nombre de n-grammes', fontsize = 10)
plt.grid()
plt.legend()
plt.savefig('bi-tri-4_sentences_10mil.png')
plt.show()



plt.plot(sentences_taken_bi[0:-3], time_bi[0:-3], '-', color = '#FF0000', label = "Bigrammes")
plt.plot(sentences_taken_tri, time_tri, '-', color = '#1e4ff1', label = "Trigrammes")
plt.plot(sentences_taken_4, time_4, '-', color = '#32CD32', label = "4-grammes")
#plt.xticks([16, 30, 50, 100, 200, 220, 240, 260, 280,  300, 400, 500,600] )
plt.xticks(fontsize=9)
plt.xticks(np.arange(0e6, 9e6, 1e6))
plt.ticklabel_format(style='sci', axis='x', scilimits=(6,6))
plt.xlabel('Nombre de phrases', fontsize = 10)
plt.ylabel('Temps [s]', fontsize = 10)
plt.grid()
plt.legend()
plt.savefig('time_sentences_8mil.png')
plt.show()
