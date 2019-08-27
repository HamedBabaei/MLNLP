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

from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV

#EVALUATION METRICS
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

#clean punchuations and stop words from txt
def Preprocessing(text , _stopwords):
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
    return cleaned_text

#return wor2vec vectors from trained word2vec model
def buildWordVector(imdb_w2v, text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

#Run tf idf with stop words or without stop words
def W2V( all_candidates_txts , all_unknowns_txts, all_truths , stopwords_list , _clf , merge_candidates, thereshold ):
    results = []
    _TP = 0 # to calculate overall TP
    _test_size = 0 # to calculate overall tested documents
    _F1_score = 0
    for problem in all_candidates_txts.keys():
        print("Working on Problem:::" , problem)
        results.append("Working on Problem :::: " + problem ) 
        results.append("                   :::: " + all_truths[problem]['language'] )
        candidates = all_truths[problem]["candidates_id"]
        results.append("                   :::: " + str(len(candidates) - 1) + " candidates")

        #prepare Train Set
        train_set = []
        train_labels = []
        for candidate , candidate_texts in all_candidates_txts[problem].items():
            if merge_candidates:
                train_set.append(candidate_texts)
                train_labels.append(candidate)
            else:
                for text in candidate_texts:
                    train_set.append(text)
                    train_labels.append(candidate)

        #prepare Test Set
        test_set = []
        test_labels = []
        index = 0
        for unknown , text in all_unknowns_txts[problem].items():
            test_set.append(text)
            test_labels.append(all_truths[problem]['truth'][index + 1])
            index += 1
    
        results.append("                   :::: " + str(len(test_labels)) + " test size")
        #Training word2vec on test and train set for classification and prediction
        #Preprocessing train and test
        train_set = [Preprocessing(text , stopwords_list[all_truths[problem]['language']]) 
                        for text in train_set]
        test_set = [Preprocessing(text , stopwords_list[all_truths[problem]['language']])
                        for text in test_set]
        #Train word2vec model on train set
        n_dim = 300
        word2vec_model = Word2Vec(sg=0, size=n_dim, min_count=1, workers=7)
        word2vec_model.build_vocab(train_set)
        for epoch in range(20):
            word2vec_model.train(train_set ,total_examples=word2vec_model.corpus_count,
                                        epochs=word2vec_model.iter)
        #Train word2vec model on train set
        for epoch in range(20):
            word2vec_model.train(test_set ,total_examples=word2vec_model.corpus_count,
                                        epochs=word2vec_model.iter)
        #prepare get train and test set vectors
        train = np.concatenate([buildWordVector(word2vec_model, text , n_dim) for text in train_set])
        train = scale(train)
        test = np.concatenate([buildWordVector(word2vec_model, text , n_dim) for text in test_set])
        test = scale(test)

        #run clf on train set
        max_abs_scaler = preprocessing.MaxAbsScaler()
        scaled_train_data = max_abs_scaler.fit_transform(train)
        scaled_test_data = max_abs_scaler.transform(test)
        clf = CalibratedClassifierCV(_clf)
        clf.fit(scaled_train_data, train_labels)
        predictions = clf.predict(scaled_test_data)
        proba = clf.predict_proba(scaled_test_data)
        # Reject option (used in open-set cases)
        count=0
        for i,p in enumerate(predictions):
            sproba=sorted(proba[i],reverse=True)
            if sproba[0] - sproba[1] < thereshold:
                predictions[i]= '<UNK>'
                count=count+1

        #Calculating F1-Macro 
        y_predict = [candidates[predict] for predict in predictions] 
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
    results.append('TP overall score ::: ' + str(_TP ))
    results.append("F1-SCORE overall ::: " + str(_F1_score / 20))
    results.append('TEST SIZE overall documents ::: ' + str(_test_size ))
    return results

def Run(all_candidates_txts , all_unknowns_txts, all_truths , stopwords_list , _clf, merge_candidates , thereshold):
    return W2V(all_candidates_txts , all_unknowns_txts, all_truths , stopwords_list , _clf , merge_candidates , thereshold)
    