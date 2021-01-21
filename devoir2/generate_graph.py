import kenlm
import os
import io
import nltk
import time
import csv
import re
import pandas as pd
import numpy as np
from nltk.tokenize import ToktokTokenizer
from string import punctuation
import matplotlib.pyplot as plt

TRAINING_FOLDER_PATH = "C:/Users/Euge/Documents/Session1-Maitrise/IFT6285-NLP/training-monolingual.tokenized.shuffled"
TEST_FOLDER_PATH = "C:/Users/Euge/Documents/Session1-Maitrise/IFT6285-NLP/heldout-monolingual.tokenized.shuffled"

tranches10_path = "C:/Users/Euge/OneDrive/Documents/Session1/IFT6285-NLP/devoir2/training_text_file/10tranches.txt"
test_path = "C:/Users/Euge/OneDrive/Documents/Session1/IFT6285-NLP/devoir2/first10000sentences_test.txt"


def write_in_file():
    """
    
    """
    # Get the files 
    list_of_files  = os.listdir(TRAINING_FOLDER_PATH)
    number_of_file = len(list_of_files)

    # Initialise the lemmatizer
    lemmatizer = nltk.WordNetLemmatizer()

    # Initialise the tokenizer
    tokenizer = ToktokTokenizer()
    tokenizer.AMPERCENT = re.compile('& '), '& '
    tokenizer.TOKTOK_REGEXES = [(regex, sub) if sub != '&amp; ' else (re.compile('& '), '& ') for (regex, sub) in
                                ToktokTokenizer.TOKTOK_REGEXES]
    toktok = tokenizer.tokenize

    total_text = pd.Series([])

    # loop in the files 
    for i in range(0,11):

        file_name = list_of_files[i]
        print(i)

        # open the files 
        with open(os.path.join(TRAINING_FOLDER_PATH, file_name), 'r', encoding="utf8") as text:

            string_text = text.read()
            splitted = toktok(string_text)
            # Lemmatize
            lemmatized = [lemmatizer.lemmatize(t) for t in splitted]
            tokens = pd.Series(lemmatized)
            # Take off random punctuation
            # All the numbers under the same name
            tokens.loc[tokens.apply(lambda x: x.isnumeric())] = "NUMBER"

            total_text = total_text.append(tokens)

    # Write in a file
    txtfilename = "training_text_file/" + str(i+1) + "yo.txt"

    with io.open(txtfilename, "w", encoding="utf-8") as f:
        for item in total_text:
            f.write("%s " % item)


def lemmatize_10tranches(path, new_name):
    """
    Clean the 10 tranches texts by lemmatize and take off punctuations and regroup the numbers
    """

    # Initialise the lemmatizer
    lemmatizer = nltk.WordNetLemmatizer()

    # Initialise the tokenizer
    tokenizer = ToktokTokenizer()
    tokenizer.AMPERCENT = re.compile('& '), '& '
    tokenizer.TOKTOK_REGEXES = [(regex, sub) if sub != '&amp; ' else (re.compile('& '), '& ') for (regex, sub) in
                                ToktokTokenizer.TOKTOK_REGEXES]
    toktok = tokenizer.tokenize

    # open the files 
    with open(path, 'r', encoding="utf8") as text:

        string_text = text.read().lower()
        splitted = toktok(string_text) 

        # Lemmatize
        lemmatized = [lemmatizer.lemmatize(t) for t in splitted]
        tokens = pd.Series(lemmatized)
        # Take off random punctuation
        # All the numbers under the same name
        tokens.loc[tokens.apply(lambda x: x.isnumeric())] = "NUMBER"

        # Write in a file
        txtfilename = new_name

    with io.open(txtfilename, "w", encoding="utf-8") as f:
        for item in tokens:
                f.write("%s " % item)


def graph_time():
    """
    Create the graph of the time to create the models
    """
    # Extracts the data
    file_   = pd.read_csv("time_and_size_2.csv", header=None).dropna()

    size_ = file_.iloc[:,0].astype(float)
    time_ = file_.iloc[:,1].astype(float)
    index = np.arange(1,len(size_)+1)

    # Plot the data
    plt.plot(index, time_, '-', color = '#4B0082')
    plt.xticks(np.arange(1, 33, 2))
    plt.grid()
    plt.xlabel('Nombre de tranches', fontsize = 10)
    plt.ylabel(r'Temps $[s]$', fontsize = 10)
    plt.show()


    plt.plot(index, size_/1.25e8, '-', color='#4B0082')
    plt.xlabel('Nombre de tranches', fontsize = 10)
    plt.ylabel(r'Taille $[Gb]$', fontsize = 10)
    plt.xticks(np.arange(1, 33, 2))
    plt.grid()
    plt.show()



def graph_perplexity():
    """
    Create the graph of the time it takes to count the number of words 
    """
    # Extracts the data
    file_   = pd.read_csv("REAL_min_max_moy_perplexity_10000_bigram.csv", header=None).dropna()

    min_ = file_.iloc[:,0].astype(float)
    max_ = file_.iloc[:,1].astype(float)
    avg_ = file_.iloc[:,2].astype(float)
    index = np.arange(1,len(min_)+1)

    print(max(min_))
    print(max(max_))
    print(max(avg_))
    print(min(avg_))

    # Plot the data
    plt.plot(index, max_, '-', color = '#FF0000')
    plt.xticks(np.arange(1, 33, 2))
    plt.grid()
    plt.xlabel('Nombre de tranches', fontsize = 10)
    plt.ylabel(r'Perplexité maximale', fontsize = 10)
    plt.show()


    plt.plot(index, min_, '-', color = '#0000FF')
    plt.xticks(np.arange(1, 33, 2))
    plt.grid()
    plt.xlabel('Nombre de tranches', fontsize = 10)
    plt.ylabel(r'Perplexité minimum', fontsize = 10)
    plt.show()


    plt.plot(index, avg_, '-', color = '#19a74a')
    plt.xticks(np.arange(1, 33, 2))
    plt.grid()
    plt.xlabel('Nombre de tranches', fontsize = 10)
    plt.ylabel(r'Perplexité moyenne', fontsize = 10)
    plt.show()
