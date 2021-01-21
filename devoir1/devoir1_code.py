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

FOLDER_PATH = "C:/Users/Euge/OneDrive/Documents/Session1/IFT6285-NLP/devoir1/training-monolingual.tokenized.shuffled" 

def measure_time_and_total_words(filename):
    """
    Count the number total words and the time it takes to measure 
    write the results in a file
    """

    # get all the files names
    list_of_files  = os.listdir(FOLDER_PATH)
    number_of_file = len(list_of_files)

    # Initialise the list 
    number_each_file      = [None] * number_of_file
    time_each_file        = [None] * number_of_file
    total_words_each_file = [None] * number_of_file
    # Initialise the accumulated time
    accumulated_time = 0
    # Initialise the tokenizer
    tokenizer = ToktokTokenizer()
    tokenizer.AMPERCENT = re.compile('& '), '& '
    tokenizer.TOKTOK_REGEXES = [(regex, sub) if sub != '&amp; ' else (re.compile('& '), '& ') for (regex, sub) in
                                ToktokTokenizer.TOKTOK_REGEXES]
    toktok = tokenizer.tokenize

    # Loop throught the files 
    for i in range(number_of_file):
        start = time.time()
        file_name = list_of_files[i]
        time_one_text = 0

        # Open the file 
        with open(os.path.join(FOLDER_PATH, file_name), 'r', encoding="utf8") as text:
            string_text = text.read()
            #tokens = string_text.split()
            #tokens = nltk.word_tokenize(string_text)
            tokens = toktok(string_text)

            end = time.time()
        
        time_one_text = end - start
        accumulated_time += time_one_text

        # Fill the list of data
        number_each_file[i]      = i
        total_words_each_file[i] = len(tokens)
        time_each_file[i]        = accumulated_time

    # Write the results in a file
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(zip(number_each_file, total_words_each_file, time_each_file))


def count_unique_words(filename, tokeniz, preprocessed = True):
    """
    Count the number of unique words with or without preprocessing and using 
    nltk toktok tokennizer or nltk.word_tokenize
    write the results in a file
    """

    # Get the files 
    list_of_files  = os.listdir(FOLDER_PATH)
    number_of_file = len(list_of_files)

    number_each_file         = [None] * number_of_file
    accumulated_unique_words = [None] * number_of_file

    # Initialise the pandas serie for unique words 
    unique_tokens = pd.Series([])
    # Initialise the lemmatizer
    lemmatizer = nltk.WordNetLemmatizer()
    # Initialise the tokenizer
    tokenizer = ToktokTokenizer()
    tokenizer.AMPERCENT = re.compile('& '), '& '
    tokenizer.TOKTOK_REGEXES = [(regex, sub) if sub != '&amp; ' else (re.compile('& '), '& ') for (regex, sub) in
                                ToktokTokenizer.TOKTOK_REGEXES]
    toktok = tokenizer.tokenize

    # loop in the files 
    for i in range(number_of_file):

        file_name = list_of_files[i]

        # open the files 
        with open(os.path.join(FOLDER_PATH, file_name), 'r', encoding="utf8") as text:

            if(preprocessed):
                string_text = text.read().lower()

                if(tokeniz == "tok"):
                    splitted = toktok(string_text)
                else:
                    splitted = nltk.word_tokenize(string_text)

                # Lemmatize
                lemmatized = [lemmatizer.lemmatize(t) for t in splitted]
                tokens = pd.Series(lemmatized)
                # Take off random punctuation
                tokens = tokens[-tokens.isin(list(punctuation))]
                # All the numbers under the same name
                tokens.loc[tokens.apply(lambda x: x.isnumeric())] = "NUMBER"
            
            # No preprocessing
            else:
                string_text = text.read()
                if(tokeniz == "tok"):
                    splitted = toktok(string_text)
                else:
                    splitted = nltk.word_tokenize(string_text)
                tokens = pd.Series(splitted)

            # Get the unique tokens
            unique_tokens = pd.Series(pd.unique(unique_tokens.append(tokens)))

        number_each_file[i] = i
        accumulated_unique_words[i] = len(unique_tokens)

    # Write in a file
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(zip(number_each_file, accumulated_unique_words))


def graph_time():
    """
    Create the graph of the time it takes to count the number of words 
    """
    # Extracts the data
    nltk_file   = pd.read_csv("wordscount_accumulatedtime_nltk.csv", header=None).dropna()
    split_file  = pd.read_csv("wordscount_accumulatedtime_split.csv", header=None).dropna()
    toktok_file = pd.read_csv("wordscount_accumulatedtime_toktok.csv", header=None).dropna()

    time_nltk   = nltk_file.iloc[:,2].astype(int)
    time_split  = split_file.iloc[:,2].astype(int)
    time_toktok = toktok_file.iloc[:,2].astype(int)
    index       = nltk_file.iloc[:,0].astype(int).add(1)

    # Plot the data
    plt.plot(index, time_nltk, '-', color = '#FF0000', label = "nltk.word_tokenize()")
    plt.plot(index, time_split, '-', color='#0000FF', label = "split()")
    plt.plot(index, time_toktok, '-', color='#19a74a', label = "ToktokTokenizer()")

    plt.xticks(np.arange(1, 109, 9))

    plt.legend()
    plt.grid()

    plt.xlabel('Nombre de tranches', fontsize = 10)
    plt.ylabel(r'Temps $[s]$', fontsize = 10)

    plt.show()


def unique_words_graph():
    """
    Create the graph of the time it takes to count the number of words 
    """
    # Extracts the data
    preproc_file  = pd.read_csv("uniquewords_toktok_preproc.csv", header=None).dropna()
    nopreproc_file  = pd.read_csv("uniquewords_toktok_NOpreproc.csv", header=None).dropna()

    unique_preproc  = preproc_file.iloc[:,1].astype(int)
    unique_nopreproc  = nopreproc_file.iloc[:,1].astype(int)
    index = preproc_file.iloc[:,0].astype(int).add(1)
    
    # Plot the data
    plt.plot(index, unique_preproc, '-', color = '#FF0000', label = "preprocessed")
    plt.plot(index, unique_nopreproc, '-', color = '#0000FF', label = "not proprocessed")

    plt.xticks(np.arange(1, 109, 9))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.grid()
    plt.legend()

    plt.xlabel('Nombre de tranches', fontsize = 10)
    plt.ylabel('Nombre de mots uniques', fontsize = 10)

    plt.show()


def unique_and_total_graph():
    """
    Create the graph of total words and unique words 
    """

    # Extracts the data
    preproc_file  = pd.read_csv("uniquewords_toktok_preproc.csv", header=None).dropna()
    toktok_file = pd.read_csv("wordscount_accumulatedtime_toktok.csv", header=None).dropna()

    unique_preproc  = preproc_file.iloc[:,1].astype(int)
    total_words_cumu = toktok_file.iloc[:,3].astype(int)
    index = preproc_file.iloc[:,0].astype(int).add(1)

    # Plot the data
    plt.plot(index, unique_preproc, '-', color = '#FF0000', label = "Mots uniques")
    plt.plot(index, total_words_cumu, '-', color = '#0000FF', label = "Mots totals")

    plt.ylim(0, 2e8)
    plt.xticks(np.arange(1, 109, 9))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend()

    plt.xlabel('Nombre de tranches', fontsize = 10)
    plt.ylabel('Nombre de mots', fontsize = 10)

    plt.show()
