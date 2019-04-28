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

import word2vec_clf as w2v
import tf_idf_clf as tfidf
import doc2vec_clf as d2v

#return Json file
def Read_json(path):
    ''' Read and return JSON file context from provided Path to the Json file!'''
    with codecs.open( path , 'r' , encoding="utf-8") as f:
        return json.load(f)

#return text file
def Read_text(path):
    ''' Read and return Text file context from provided Path to the Text file!'''
    with codecs.open( path , 'r' , encoding='utf-8') as f:
        return f.read()

def Write_text(path , text):
    with codecs.open( path , 'w' , encoding='utf-8') as f:
        f.write(text)

#return all_candidates_txts , all_unknowns_txts, all_truths
def Read_problems(dataset_root_dir , merge_candidates):
    all_unknowns_txts = {}
    all_candidates_txts = {}
    all_truths = {}
    
    #read dataset info
    problems = Read_json(os.path.join( dataset_root_dir , "collection-info.json"))
    
    #walk trough each problem
    for problem in problems:
        print("Loading problem : " , problem['problem-name'])
        
        # read problem info
        problem_info = Read_json(os.path.join( dataset_root_dir , problem['problem-name'], 'problem-info.json'))
        
        #read candidates txt of the problem
        candidates = [candidate['author-name'] for candidate in problem_info['candidate-authors']]
        candidates_txts = {}
        for candidate in candidates:
            txt = []
            for txt_name in os.listdir(os.path.join( dataset_root_dir , problem['problem-name'] , candidate)):
                txt_path = os.path.join(os.path.join( dataset_root_dir , problem['problem-name'] , candidate , txt_name))
                txt.append(Read_text(txt_path))
            if merge_candidates:
                candidates_txts[candidate] = ' '.join(txt)
            else:
                candidates_txts[candidate] = txt
        all_candidates_txts[problem['problem-name']] = candidates_txts
        
        #read unknowns txt of the problem
        unknowns_root_dir = os.path.join( dataset_root_dir , problem['problem-name'] , "unknown")
        unknowns_txts = {}
        for unknown in os.listdir(unknowns_root_dir):
            unknown_txt = Read_text(os.path.join(unknowns_root_dir , unknown))
            unknowns_txts[unknown] = unknown_txt
        all_unknowns_txts[problem['problem-name']] = unknowns_txts
        
        #read truth of the unknowns txt of the problem
        truth , label = {} , {}
        candidates = {'<UNK>':0}
        for index in range(0 , len(problem_info['candidate-authors'])):
            candidates[problem_info['candidate-authors'][index]['author-name']] = index + 1
        ground_truth = Read_json(os.path.join(dataset_root_dir , problem['problem-name'], 'ground-truth.json'))
        
        for index in range(0,len(ground_truth['ground_truth'])):
            label[index + 1]= candidates[ground_truth['ground_truth'][index]['true-author']]

        truth["language"] = problem['language']
        truth["candidates"] = len(candidates) - 1
        truth["truth"] = label
        truth["candidates_id"] = candidates
        all_truths[problem['problem-name']] = truth
        
    return all_candidates_txts , all_unknowns_txts, all_truths

def main(input , output , merge_candidates ):
    stopwords_list = {'en': set(stopwords.words('english')) , 'fr':set(stopwords.words('french')),
                      'sp': set(stopwords.words('spanish')) , 'it':set(stopwords.words('italian'))}
    all_candidates_txts , all_unknowns_txts, all_truths = Read_problems(dataset_root_dir = input , merge_candidates = merge_candidates)
    classifiers = { "LogisticRegression" : LogisticRegression(),
                    #"LinearSVC":LinearSVC(),
                    #"RandomForestClassifier": RandomForestClassifier(),
                    #"DecisionTreeClassifier": DecisionTreeClassifier(),
                    #"MLPClassifier": MLPClassifier(),
                    #"BernoulliNB": BernoulliNB(),
                    #"GaussianNB":GaussianNB(),
                    #"SVC(rbf)":SVC(),
                    #"SGD": SGDClassifier() ,
                    #"KNN":KNeighborsClassifier(), 
                    }
    item = 0
    for classifier_name , classifier in classifiers.items():

        item += 1
        print("working on " , str(item) + '- ' + classifier_name + ' + ' + 'Word2Vec')
        results = w2v.Run(all_candidates_txts , all_unknowns_txts, all_truths , stopwords_list , classifier,  merge_candidates)
        path = os.path.join(output , str(item) + '- ' + classifier_name + ' + ' + 'Word2Vec.txt')
        Write_text(path , '\n'.join(results) )
        
        item += 1
        print("working on " , str(item) + '- ' + classifier_name + ' + ' + 'Doc2Vec')
        results = d2v.Run(all_candidates_txts , all_unknowns_txts, all_truths , stopwords_list , classifier , merge_candidates)
        path = os.path.join(output , str(item) + '- ' + classifier_name + ' + ' + 'Doc2Vec.txt')
        Write_text(path , '\n'.join(results) ) 
        
        item += 1
        print("working on " , str(item) + '- ' + classifier_name + ' + ' + 'TF-IDF')
        results = w2v.Run(all_candidates_txts , all_unknowns_txts, all_truths , stopwords_list , classifier, merge_candidates)       
        path = os.path.join(output , str(item) + '- ' + classifier_name + ' + ' + 'TFIDF.txt')
        Write_text(path , '\n'.join(results) ) 

main( input = "cross_dataset" , output = "out" , merge_candidates = False)