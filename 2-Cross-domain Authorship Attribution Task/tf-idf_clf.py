import os
import nltk
import json
import codecs
import gensim
from nltk.tokenize import word_tokenize
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import re
from nltk.stem import WordNetLemmatizer
import string
from sklearn.metrics import classification_report


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

#calculating Evaluation Metrics
def Precision(TP , TN , FP , FN):
    return TP /(TP + FP)

def Recall(TP , TN , FP , FN):
    return TP / (TP + FN)

def F1(recall, precision):
    if recall == 0 and precision == 0:
        return 0
    return 2 * (recall * precision) / (recall + precision)

def Accuracy(TP , TN , FP , FN):
    return (TP + TN) / (TP + FP + FN + TN)

#calculate confusion matrix based on normalized problem truth and predictions 
# and returns  TP, FP, FN, TN
def Confusion_matrix(truth_problem, prediction):
    TP, TN, FP, FN = 0, 0, 0, 0
    for problem_id, problem_value in truth_problem.items():
        for truth_id, truth_value in problem_value.items():
            for i in range(10):
                if i == prediction[problem_id][truth_id] and prediction[problem_id][truth_id] == problem_value[truth_id]:
                    TP += 1
                elif i == prediction[problem_id][truth_id] and prediction[problem_id][truth_id] != problem_value[truth_id]:
                    FP += 1
                elif i != prediction[problem_id][truth_id] and i == problem_value[truth_id]:
                    FN += 1
                elif i != prediction[problem_id][truth_id] and i != problem_value[truth_id]:
                    TN += 1
    return TP, FP, FN, TN

#Normalizing the truths of the problem
def Normal_context(all_truth_context, **args):
    truth_problem = {}
    if args['whole_documents'] == 'on':
         for key, values in all_truth_context.items():
                truth_problem[key] = values['truth']

    elif args['whole_documents'] == 'off' and 'target_problem' not in args.keys():
        raise TypeError('Expected the value of the target_problem assigned to some value')

    elif args['whole_documents'] == 'off':  
        truth_problem[args['target_problem']] = all_truth_context[args['target_problem']]['truth']
    return truth_problem

#return all_candidates_txts , all_unknowns_txts, all_truths
def Read_problems(dataset_root_dir , merge_candidates = False):
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

#prediction for classifiers
def Prediction(clf , test_tfidf , labels , candidates):
    prediction_dict = {}
    for index in range(0 , len(labels)):
        predict = clf.predict(test_tfidf[index])[0]
        prediction_dict[index+1] = candidates[predict]
    return prediction_dict

#clean punchuations and stop words from txt
def preprocessing(text):
    lemmatizer = WordNetLemmatizer()
    cleaned_text = []
    text.replace('-' , '')
    text.replace('.' , '')
    for word in text.split():
        cleaned_word = [letter for letter in word if letter not in string.punctuation]
        cleaned_text.append(lemmatizer.lemmatize(''.join(cleaned_word).lower()))
    return ' '.join(cleaned_text)

#Run tf idf with stop words or without stop words
def main(tfidf_max_features = 200 , with_stop_words = False , clean_text = False ):
    _F1 = 0 # to calculate overall f1
    _TP = 0 # to calculate overall TP
    _test_size = 0 # to calculate overall tested documents

    dataset_root_dir = "cross_dataset" #path to the dataset root
 
    all_candidates_txts , all_unknowns_txts, all_truths = Read_problems(dataset_root_dir , merge_candidates = True)
    print('----------------------------------------------')
    for test_problem_name in all_candidates_txts.keys():
        print("Working on Problem :::: " , test_problem_name ) # print problem name
        print("                   :::: " , all_truths[test_problem_name]['language'] )
        candidates = all_truths[test_problem_name]["candidates_id"]
        print("                   :::: " , len(candidates) - 1 , " candidates")

        #prepare Train Set
        train_set = []
        train_labels = []
        for candidate , text in all_candidates_txts[test_problem_name].items():
            if clean_text:
                train_set.append(preprocessing(text))
            else:
                train_set.append(text)
            train_labels.append(candidate)

        #prepare Test Set
        test_set = []
        test_labels = []
        index = 0
        for unknown , text in all_unknowns_txts[test_problem_name].items():
            if clean_text:
                test_set.append(preprocessing(text))
            else:
                test_set.append(text)
            test_labels.append(all_truths[test_problem_name]['truth'][index + 1])
            index += 1

        print("                   :::: " , len(test_labels) , " test size")

        #Initialize TF-IDF
        if with_stop_words:
            vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
        else:
            # this stopwords must be change for each 5 languages
            vectorizer = TfidfVectorizer(max_features=tfidf_max_features , stop_words='english' )
        #Train tf-idf on train set
        train_tfidf = vectorizer.fit_transform(train_set)
        print("                   :::: Actual number of tfidf features on Train set: %d" % train_tfidf.get_shape()[1])
        #Train tf-idf on test set
        test_tfidf = vectorizer.fit_transform(test_set)
        print("                   :::: Actual number of tfidf features on Test set: %d" % test_tfidf.get_shape()[1])

        #run clf on train set
        clf = LinearSVC()
        clf.fit(train_tfidf , train_labels)
        #make predictions on test set

        predicts_dict = predicts_dict = Prediction( clf , test_tfidf , test_labels , candidates)

        #Evaluate
        predictions = {test_problem_name:{'language':all_truths[test_problem_name]['language'] 
                        , 'truth':predicts_dict , 'candidates':len(candidates) - 1 }}
        truth_problems = Normal_context(all_truths , whole_documents = "off" , target_problem = test_problem_name)
        predictions_normalize = Normal_context(predictions , whole_documents = "off",  target_problem = test_problem_name)
        TP , FP , FN , TN = Confusion_matrix(truth_problems, predictions_normalize)
        accuracy = Accuracy(TP , TN , FP , FN)
        recall = Recall(TP , TN , FP , FN)
        precision = Precision(TP , TN , FP , FN)
        f1 = F1(recall  , precision)
        print("                   ::::  F1    " , f1)
        print("                   ::::  TP    " , TP)
        _F1 = _F1 + f1
        _TP = _TP + TP
        _test_size = _test_size + len(test_labels)
        print('----------------------------------------------')
    print("OVERALL RESULTS  ::: ")
    print("TF-IDF Maximum Features ::: " , tfidf_max_features )
    print("STOP WORDS IN TF-IDF USAGE ::: " , str(with_stop_words))
    print('F1 overall score ::: ' , _F1/20 )
    print('TP overall score ::: ' , _TP )
    print('TEST SIZE overall documents ::: ' , _test_size )

main(clean_text=True)