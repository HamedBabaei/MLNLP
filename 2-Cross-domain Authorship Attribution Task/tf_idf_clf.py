import os
import json
import codecs
import operator
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.preprocessing import scale
from tqdm import tqdm
from sklearn import utils
#MODELS
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
#CLASSIFIERS
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC ,SVC
from sklearn.neural_network import MLPClassifier 
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier 

#EVALUATION METRICS
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


#prediction for classifiers
def Prediction(clf , test_tfidf , labels , candidates):
    prediction_dict = {}
    for index in range(0 , len(labels)):
        predict = clf.predict(test_tfidf[index])[0]
        prediction_dict[index+1] = candidates[predict]
    return prediction_dict

#clean punchuations and stop words from txt
def preprocessing(text , _stopwords):
    lemmatizer = WordNetLemmatizer()
    cleaned_text = []
    text = text.replace('-' , '')
    text = text.replace('.' , '')
    text = text.replace('”' , '')
    text = text.replace('’' , '')
    text = text.replace('“' , '')
    text = text.replace('‘' , '')
    for word in text.split():
        if word not in _stopwords:
            cleaned_word = [letter for letter in word if letter not in string.punctuation]
            cleaned_text.append(lemmatizer.lemmatize(''.join(cleaned_word).lower()))
    return ' '.join(cleaned_text)

#Run tf idf with stop words or without stop words
def TFIDF( all_candidates_txts , all_unknowns_txts, all_truths , stopwords_list, clf , 
        tfidf_max_features = 200 , with_stop_words = False , clean_text = False ):
    results = []
    _TP = 0 # to calculate overall TP
    _test_size = 0 # to calculate overall tested documents
    _F1_score = 0
    language = {'en': 'english', 'fr':'french', 'sp': 'spanish', 'it':'italian'}
    for problem in all_candidates_txts.keys():
        results.append("Working on Problem :::: " + problem ) # print problem name
        results.append("                   :::: " + all_truths[problem]['language'] )
        candidates = all_truths[problem]["candidates_id"]
        results.append("                   :::: " + str(len(candidates) - 1) + " candidates")

        #prepare Train Set
        train_set = []
        train_labels = []
        for candidate , text in all_candidates_txts[problem].items():
            if clean_text:
                train_set.append(preprocessing(text , stopwords_list))
            else:
                train_set.append(text)
            train_labels.append(candidate)

        #prepare Test Set
        test_set = []
        test_labels = []
        index = 0
        for unknown , text in all_unknowns_txts[problem].items():
            if clean_text:
                test_set.append(preprocessing(text , stopwords_list))
            else:
                test_set.append(text)
            test_labels.append(all_truths[problem]['truth'][index + 1])
            index += 1

        results.append("                   :::: " + str(len(test_labels)) + " test size")

        #Initialize TF-IDF
        if with_stop_words:
            vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
        else:
            # this stopwords must be change for each 5 languages
            vectorizer = TfidfVectorizer(max_features=tfidf_max_features , 
                                        stop_words=language[all_truths[problem]['language']] )
        #Train tf-idf on train set
        train_tfidf = vectorizer.fit_transform(train_set)
        results.append("                   :::: Actual number of tfidf features on Train set: " + str(train_tfidf.get_shape()[1]))
        #Train tf-idf on test set
        test_tfidf = vectorizer.fit_transform(test_set)
        results.append("                   :::: Actual number of tfidf features on Test set: " + str(test_tfidf.get_shape()[1]))

        #run clf on train set
        clf.fit(train_tfidf , train_labels)
        #make predictions on test set

        predicts_dict = Prediction( clf , test_tfidf , test_labels , candidates)

        y_predict = [predict for _ , predict in predicts_dict.items()] 
        _f1_score = f1_score(test_labels , y_predict , average = 'macro')
        TP = sum([ 1 for i in range(0 , len(test_labels)) if y_predict[i] == test_labels[i]  ])
        results.append("                   :::: F1    " + str(_f1_score))
        results.append("                   :::: TP    " + str( TP))
        results.append("  Classification Report:\n" + str(classification_report(test_labels ,y_predict)) + "\n")
        results.append('--------------------------------------------------------------')
        _test_size = _test_size + len(test_labels)
        _F1_score = _F1_score + _f1_score
        _TP = _TP + TP


    results.append("OVERALL RESULTS  ::: ")
    results.append("TF-IDF Maximum Features ::: " + str(tfidf_max_features ))
    results.append("STOP WORDS IN TF-IDF USAGE ::: " + str(with_stop_words))
    results.append('TP overall score ::: ' + str(_TP) )
    results.append("F1-SCORE overall ::: " + str(_F1_score / 20))
    results.append('TEST SIZE overall documents ::: ' + str(_test_size ))
    return results

def Run(all_candidates_txts , all_unknowns_txts, all_truths , stopwords_list , clf ):
    return TFIDF(all_candidates_txts , all_unknowns_txts, all_truths , stopwords_list , clf)
    