import re
import pandas as pd
import numpy as np
from nltk.tokenize import ToktokTokenizer
from string import punctuation
import matplotlib.pyplot as plt
import tarfile
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import spacy
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import PathLineSentences
import multiprocessing
from sklearn.neighbors import DistanceMetric
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier


from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
import string
# import logging

# prog = 'train_model'
# logname = f"/grid_search.log"
# logging.basicConfig(  # filename=logname,
#     format="%(levelname)s %(asctime)s [" + \
#     prog + "] %(message)s",
#     datefmt='%H:%M:%S', level=logging.INFO,
#     handlers=[
#         logging.FileHandler(logname),
#         logging.StreamHandler()
#     ])

# logging.info(
#     f"running with config: {ext} -- hopefully generating {modelname} {txt_modelname}")


nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words("english")
wordnet = WordNetLemmatizer()
p_stemmer = PorterStemmer()


def print_scores(actual, predicted, averaging_type):
    print('\nAVERAGING TYPE==> ', averaging_type)
    print('F1 score: ', f1_score(actual, predicted, average=averaging_type))
    print('Average Precision Score: ', average_precision_score(
        actual, predicted, average=averaging_type))
    print('Average Recall Score: ', recall_score(
        actual, predicted, average=averaging_type))


def get_data(name_train, name_test_closed, name_test_open):
    PATH_TRAINING = name_train
    PATH_TEST = name_test_open

    train_set = pd.read_csv(PATH_TRAINING, header=0, sep=',', quotechar='"', names=[
                            'autor', 'gender', 'age', 'text'])
    test_set = pd.read_csv(PATH_TEST, header=0, sep=',', quotechar='"', names=[
                           'autor', 'gender', 'age', 'text'])

    # train_set = train_set.groupby('autor').agg({'gender': 'first',
    #                                             'age': 'first',
    #                                             'text': ' '.join}).reset_index()

    punct = set(punctuation)
    punct_cleaned = set(punctuation.replace("-", "").replace("'", ""))

    for i in range(len(train_set)):
        old_text = train_set["text"].iloc[i]

        # split into sentences
        sentences = re.split('[.!?]', old_text)
        sentences_clean = []

        # for each sentences
        for sentence in sentences:
            sentence_new = []
            for word in sentence.split():
                if word not in punct_cleaned:
                    word = ''.join(
                        ch for ch in word if ch not in punct_cleaned)
                    word = word.lower()
                    word = SnowballStemmer('english').stem(word)
                    if word.isnumeric():
                        word = "NUMBER"
                    sentences_clean.append(word)
        train_set["text"].iloc[i] = ' '.join(sentences_clean)

    for i in range(len(test_set)):
        old_text = test_set["text"].iloc[i]

        # split into sentences
        sentences = re.split('[.!?]', old_text)
        sentences_clean = []

        # for each sentences
        for sentence in sentences:
            sentence_new = []
            for word in sentence.split():
                if word not in punct_cleaned:
                    word = ''.join(
                        ch for ch in word if ch not in punct_cleaned)
                    word = word.lower()
                    word = SnowballStemmer('english').stem(word)
                    if word.isnumeric():
                        word = "NUMBER"
                    sentences_clean.append(word)
        test_set["text"].iloc[i] = ' '.join(sentences_clean)

    return train_set, test_set


def embedding(train_set, test_set):
    tfidf = TfidfVectorizer(max_features=1000)  # , stop_words = "english")
    X_train = tfidf.fit_transform(train_set["text"].values).toarray()
    X_test = tfidf.transform(test_set["text"].values).toarray()

    y_train = train_set["autor"]
    y_test = test_set["autor"]

    return X_train, y_train, X_test, y_test


def measure_accuracy(prediction, truth):
    good = 0
    len_testset = len(truth)

    for i in range(len_testset):
        if(truth[i] == prediction[i]):
            good += 1
    return good / len_testset


def train_model(model, prams, X_train, y_train, X_test, y_test):
    vectorizer = TfidfVectorizer
    pipe = Pipeline([
        ('vect', vectorizer()),
        # ('scaler', MaxAbsScaler()),
        ('model', model())
    ])
    pipe_params = prams
    # Instantiate GridSearchCV.

    gs = GridSearchCV(pipe,  # what object are we optimizing?
                      param_grid=pipe_params,  # what parameters values are we searching?
                      cv=10,
                      scoring='accuracy', verbose=10,
                      n_jobs=-1)  # 5-fold cross-validation.
    # Fit GridSearch to training data.
    gs.fit(X_train, y_train)
    # Score model on training set.
    train_sc = gs.score(X_train, y_train)
    # Score model on testing set.
    test_sc = gs.score(X_test, y_test)

    preds = gs.predict(X_test)
    return gs.best_params_, gs, preds, train_sc, test_sc


def get_best_model(model, prams, X_train, y_train, X_test, y_test):
    print("Now training the model : " + model.__name__)
    best_params_, gs, preds, train_sc, test_sc = train_model(
        model, prams, X_train, y_train, X_test, y_test)

    return best_params_, gs, preds, train_sc, test_sc


def load_data(name_train, name_test_closed, name_test_open):
    print("Now loading data and spliting")
    print(name_train)
    train_set, test_set = get_data(
        name_train, name_test_closed, name_test_open)
    X_train, y_train, X_test, y_test = train_set["text"], train_set[
        "autor"], test_set["text"], test_set["autor"]

    # tfidf = TfidfVectorizer(max_features=1000) #, stop_words = "english")
    # regroup_train_set = train_set.groupby('autor').agg({'gender':'first',
    #                          'age':'first',
    #                          'text': ' '.join}).reset_index()
    # tfidf.fit_transform(regroup_train_set["text"].values).toarray()
    # X_train = tfidf.transform(train_set["text"].values).toarray()
    # X_test = tfidf.transform(test_set["text"].values).toarray()
    return X_train, y_train, X_test, y_test
