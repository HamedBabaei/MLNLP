import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import json
import codecs
import os
import data_visualization as dv
import re 
class Pan:
    english_stopwords = set(nltk.corpus.stopwords.words('english'))
    verb_list = []
    pie_list = {}
    input_file = open("mytext.txt", "w")
    indentedPos_list = []    
    #preperacessing
    def Sentence_tokenizer(self , text, **args):
        if len(args) < 1:
            args["tokenizer"] = "nltk"
        if args["tokenizer"] == "nltk":
            self.sentences_tokenize = nltk.sent_tokenize(text)
        return self.sentences_tokenize

    def Word_tokenizer(self, text, **args):
        if len(args) < 1:
            args["tokenizer"] = "nltk"
        if args["tokenizer"] == "nltk":
            self.words_tokenize = nltk.word_tokenize(text)
        else:
            pass
        return self.words_tokenize

    def Stop_words_filtering(self, text):
        tokens = word_tokenize(text)
        self.contenet_tokens_without_stopwords = [token for token in tokens if token.lower() not in self.english_stopwords]
        return self.contenet_tokens_without_stopwords 

    def Pos_tagger(self, **args):
        if len(args) < 1:
            args["pos_tagger"] = "nltk"
        if args["pos_tagger"] == "nltk":
            self.token_pos_tag = pos_tag(self.words_tokenize)
        return self.token_pos_tag
      
    def Terms_counter(self, **args):
        print(".......................................................................................................................................................")
        for id, pos in args.items():
            self.Pos_counter(pos, self.verb_list)
            self.pie_list[pos] = len(self.verb_list)
            print(self.verb_list)
            #self.input_file.write(" ".join(self.verb_list))
            for word in self.verb_list:
                self.input_file.write(str(word) + "\n")
            print(".......................................................................................................................................................")

        self.Pie(self.pie_list)
      
    def Pos_counter(self, pos, pos_list, uniqeWords = False):
        pos_list.clear()
        for word in self.token_pos_tag:
            if uniqeWords:
                if word[1] == pos and word not in pos_list:
                    pos_list.append(word)
            else:
                if word[1] == pos:
                    pos_list.append(word)
    
    def Comperation_of_two_documents(self, document1, document2,**args):
        words_tokenize = nltk.word_tokenize(document1)
        word_tokenize2 = nltk.word_tokenize(document2)
        self.token_pos_tag = nltk.pos_tag(words_tokenize)
        self.Terms_counter(**args)
        self.token_pos_tag = nltk.pos_tag(word_tokenize2)
        self.Terms_counter(**args)

    
    
    #visualization
    def Pie(self, args ,**argss):
        """
        Args here:
        args = { 'x' : ['Cookies', 'Jellybean', 'Milkshake', 'Cheesecake'],
                 'y' : [38.4, 40.6, 20.7, 10.3] , 
                 'colors' : ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']}
        """
        print(args)
        x = [key for key , value in args.items()]
        y = [value for key , value in args.items()]
        colors = [ c for c in range(1 , len(args) + 1)] 
        dv.pieplot(x = x , y = y , colors = colors )
        
    def Bar(self, args ,**argss):
        """
        Args here:
        args = { 'x': ('a' , 'b' , 'c', 'd' , 'f') , 
                 'y' = [1, 2, 3 , 4 , 5] 
                }
        """
        x = ()
        for key , value in args.items():
            x += (key, ) 
        y = [ value for key , value in args.items()]
        dv.bar(x = x , y = y )#,  xlabel = args['xlabel'] if 'xlabel' in args.keys() , 
                   #ylabel=args['ylabel'] if 'ylabel' in args.keys() , title= args['title'] if 'title' in args.keys())
        
    def Scatter(self, args, **argss):
        """
        Args here:
        args = { 'x': [1 ,3 ,3 ,4 ,5 ... ] ,
                 'y': [1 ,3 ,3 ,4 ,5 ... ] ,
                 'colors': [3 , 3, 4, 5 ... ]
                }
        """
        if ('x' and 'y' and 'colors' not in args.keys()) and (type(args['x']) and type(args['y']) and type(args['colors']) is not list):
            raise TypeError("Excpected 'x' , 'y' and 'colors' list arguments doesn't provided !")
        else:
            dv.scatter( x = args['x'] , y = args['y'] , colors = args['colors'])

    def Hist(self, args, **argss):
        """
        Args Here:
        args = { 
                'data' : ( a , b , c ....) ,
                'labels' : { 'a' ,'b', 'c' , ... }
                }
        here a, b , c and .. in 'data' key values are Lists
        """
        if ('data' and 'labels' not in args.keys()) and type(args['data']) is not tuple and type(args['colors']) is not dict:
            raise TypeError("Excpected 'data' (touple of lists) and 'labels' (dict) arguments doesn't provided !")
        else:
            dv.hist( data = args['data'] , labels = args['labels'])


    #machine learning & natural language processing
    def Tf_idf(self):
        pass
    def Bag_of_words(self, **text):
        pass  


    #interpret NLP
    def Stop_words(self):
        return self.english_stopwords


    #performance
    def Precision(self):     
        self.precision = self.TP / (self.TP + self.FP)
        return self.precision

    def Recall(self):
        self.recall = self.TP / (self.TP + self.FN)
        return self.recall

    def F1(self):
        f1_score = 2 * (self.recall * self.precision) / (self.recall + self.precision)
        return f1_score

    def Accuracy(self):
        self.accuracy = (self.TP + self.TN) / (self.TP + self.FP + self.FN + self.TN)
        return self.accuracy

    def C_at_1(self):
        pass

class Cross_Domain_Authorship_Attribution(Pan):
    merge_candidates = False
    all_candidates_txts = {}
    dataset_root_dir = ""
    #dataset path must be provided

    def Read_json_file(self, json_file_name):
        with codecs.open(json_file_name, 'r', encoding= 'utf-8') as json_read:
            self.json_context = json.load(json_read)
        return self.json_context
    
 
    def Normal_context(self, **args):
        self.truth_problem = {}
        if args['whole_documents'] == 'on':
             for key, values in self.json_context.items():
                 self.truth_problem[key] = values['truth']
        elif args['whole_documents'] == 'off' and 'target_problem' not in args.keys():
            raise TypeError('Expected the value of the target_problem assigned to some value')
    
        elif args['whole_documents'] == 'off':  
            self.truth_problem[args['target_problem']] = self.json_context[args['target_problem']]['truth']
    
        return self.truth_problem
        
    def Confusion_matrix(self, prediction):
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
        f = open('confusion.txt','w')
        for problem_id, problem_value in self.truth_problem.items():
            for truth_id, truth_value in problem_value.items():
                for i in range(10):
                    if i == prediction[problem_id][truth_id] and prediction[problem_id][truth_id] == problem_value[truth_id]:
                        self.TP += 1
                    elif i == prediction[problem_id][truth_id] and prediction[problem_id][truth_id] != problem_value[truth_id]:
                        self.FP += 1
                    elif i != prediction[problem_id][truth_id] and i == problem_value[truth_id]:
                        self.FN += 1
                    elif i != prediction[problem_id][truth_id] and i != problem_value[truth_id]:
                        self.TN += 1
                    f.write('{} {} {} {}'.format(self.TP, self.TN, self.FP, self.FN))
                    f.write("\n")
                f.write('\n')
        f.close()


    def Set_dataset_dir(self , path):
        self.dataset_root_dir = path


    def Read_collection_info(self):
        with codecs.open(os.path.join(self.dataset_root_dir , "collection-info.json") , 'r' , encoding="utf-8") as f:
                return json.load(f)
    
    def Read_problems(self):
        problems = self.Read_collection_info()  
        for problem in problems:
            print("working on problem : " , problem['problem-name'])
            with codecs.open(os.path.join(self.dataset_root_dir , problem['problem-name'], 'problem-info.json') ,'r' , encoding='utf-8') as j:
                problem_info = json.load(j)
            candidates = [candidate['author-name'] for candidate in problem_info['candidate-authors']]
            candidates_txts = {}
            for candidate in candidates:
                txt = []
                for txt_name in os.listdir(os.path.join(self.dataset_root_dir , problem['problem-name'] , candidate)):
                    txt_path = os.path.join(os.path.join(self.dataset_root_dir , problem['problem-name'] , candidate , txt_name))
                    with codecs.open(txt_path , 'r' , encoding='utf-8') as f:
                        txt.append(f.read())
                if self.merge_candidates:
                    candidates_txts[candidate] = ' '.join(txt)
                else:
                    candidates_txts[candidate] = txt
                    #yield candidates_txts
            self.all_candidates_txts[problem['problem-name']] = candidates_txts
    
class Style_Change_Detection(Pan):
    pass

class Celebrity_Profiling(Pan):
    pass

class Bots_And_Gender_Profiling(Pan):
    pass

import os

def main():
    cr_d = Cross_Domain_Authorship_Attribution()
    unknown = open("known00001.txt", "r")
    known = open("unknown00001.txt", "r")
    cr_d.Comperation_of_two_documents(unknown.read(), known.read(), pos1 = "CC", pos2 = "JJ", pos3 = "VB")
    
if __name__ == "__main__": main()
