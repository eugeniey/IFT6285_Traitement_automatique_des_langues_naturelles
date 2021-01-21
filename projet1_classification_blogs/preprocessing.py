import time
import pandas as pd
import csv
import os
import string as st
import numpy as np
import itertools
from porter2stemmer import Porter2Stemmer
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import multiprocessing
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from collections import Counter
from nltk.stem.snowball import SnowballStemmer


stemmer = SnowballStemmer('english')
stopWords = stopwords.words("english")
cores = multiprocessing.cpu_count()



PATH_TRAINING = "C:/Users/Euge/Documents/Session1-Maitrise/blog_train/train/"
PATH_TEST1 = "C:/Users/Euge/Documents/Session1-Maitrise/blog_train/test/test1/"
PATH_TEST2 = "C:/Users/Euge/Documents/Session1-Maitrise/blog_train/test/test2/"
BLIND_TEST = "C:/Users/Euge/Documents/Session1-Maitrise/quiz.csv"
training_file = os.listdir(PATH_TRAINING)


# Get the training set
def saving_by_author(directory,PATH_SOURCE,destinationfile):
    data_frame=[]
    count=0
    for file in directory:
        all_text= None
        count=count+1
        path = PATH_SOURCE + file
        df = pd.read_csv(path, header=0, sep=',', quotechar='"', names=['code', 'gender', 'age', 'sign', 'text'], encoding="utf8")
        if df.empty:
            ff = open(path, encoding="utf8")
            reader = csv.reader(ff)
            for row in reader:
                df=df.append(pd.DataFrame({ 'code': row[0], 'gender': row[1],'age':row[2],'sign': row[3],'text': row[4]},index=[0]))
        all_text = df.groupby('code')['text'].apply(' '.join).reset_index()
        text = list(all_text['text'])[0]
        df_to_append=pd.DataFrame({ 'code': df['code'][0], 'gender': df['gender'][0],'age':df['age'][0],'sign': df['sign'][0],'text': text},index=[0])
        data_frame.append(df_to_append)
    new_set = pd.concat(data_frame,ignore_index=True)
    new_set.to_csv(destinationfile,index=False)
    print(len(new_set))
    return new_set

# Get the blind test set
def get_pdframe_blindtest():
    return pd.read_csv(BLIND_TEST, header=None, sep=',', quotechar='"', names=['code', 'text'])


# Get the test set
def get_pdframe_test():
    test_file1 = os.listdir(PATH_TEST1)
    test_file2 = os.listdir(PATH_TEST2)
    frames_test = []

    for file in test_file1:
        path = PATH_TEST1 + file
        df = pd.read_csv(path, header=0, sep=',', quotechar='"', names=['code', 'gender', 'age', 'sign', 'text'])
        frames_test.append(df)
        
    for file in test_file2:
        path = PATH_TEST2 + file
        df = pd.read_csv(path, header=0, sep=',', quotechar='"', names=['code', 'gender', 'age', 'sign', 'text'])
        frames_test.append(df)
    
    test_set = pd.concat(frames_test, ignore_index=True)
    
    return test_set


print("Get the data")
#apply change and save
rain_set = saving_by_author(training_file,PATH_TRAINING,'train_set_by_author.csv')
test_set = get_pdframe_test()
blind_test = get_pdframe_blindtest()


# ====================================
# ==== Preprocessing =================
# ====================================
print("Tokenise and clean the text")
def preprocessing(string_input):
    hey = [stemmer.stem(word) for word in string_input.lower().translate(str.maketrans(st.punctuation, " "*len(st.punctuation))).translate(str.maketrans('', '', st.punctuation)).split() if word not in stopWords]
    return hey


#apply tokenization
train_set['tokenized_text'] = train_set['text'].apply(preprocessing)
test_set['tokenized_text'] = test_set['text'].apply(preprocessing)
blind_test['tokenized_text'] = blind_test['text'].apply(preprocessing)

#save
train_set.to_csv('train_set_by_author_token.csv',index=False)
test_set.to_csv('test_set_token.csv',index=False)


# ====================================
# ======== To string =================
# ====================================
print("List to string")
def list_to_string(list_stem):
    return (' ').join(list_stem)


test_set['tokenized_text_string']  = test_set['tokenized_text'].apply(list_to_string)
train_set['tokenized_text_string'] = train_set['tokenized_text'].apply(list_to_string)
blind_test['tokenized_text_string'] = blind_test['tokenized_text'].apply(list_to_string)

train_set['gender'] = train_set['gender'].map({'female': 1, 'male': 0})
test_set['gender']  = test_set['gender'].map({'female': 1, 'male': 0})
train_set['sign'] = train_set['sign'].map({'Aquarius': 0, 'Aries': 1, 'Cancer': 2, 'Capricorn': 3, 
                                                'Gemini': 4, 'Leo': 5, 'Libra': 6, 'Pisces': 7, 
                                                'Sagittarius': 8, 'Scorpio': 9, 'Taurus': 10, 'Virgo': 11})

test_set['sign'] = test_set['sign'].map({'Aquarius': 0, 'Aries': 1, 'Cancer': 2, 'Capricorn': 3, 
                                                'Gemini': 4, 'Leo': 5, 'Libra': 6, 'Pisces': 7, 
                                                'Sagittarius': 8, 'Scorpio': 9, 'Taurus': 10, 'Virgo': 11})

train_set['age'] = pd.cut(train_set['age'], bins=[0,19,29,100], right=True, labels=False)
test_set['age'] = pd.cut(test_set['age'], bins=[0,19,29,100], right=True, labels=False)



train_set.to_pickle("clean_trainset.pkl")
test_set.to_pickle("clean_testset.pkl")
blind_test.to_pickle("blind_test.pkl")
