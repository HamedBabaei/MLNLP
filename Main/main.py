import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

import json
import codecs
import os
import data_visualization as dv

class Pan:
    english_stopwords = set(nltk.corpus.stopwords.words('english'))
        
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

    def Pos_tagger(self, tokens, **args):
        if len(args) < 1:
            args["pos_tagger"] = "nltk"
        if args["pos_tagger"] == "nltk":
            self.token_pos_tag = pos_tag(tokens)
        return self.token_pos_tag
            
    def Terms_counter(self, text, **args):
        pass

    #visualization
    def Pie(self, args ,**argss):
        """
        Args here:
        args = { 'x' : ['Cookies', 'Jellybean', 'Milkshake', 'Cheesecake'],
                 'y' : [38.4, 40.6, 20.7, 10.3] , 
                 'colors' : ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']}
        """
        if ('x' and 'y' and 'colors' not in args.keys()) and (type(args['x']) and type(args['y']) and type(args['colors']) is not list):
            raise TypeError("Excpected 'x' , 'y' and 'colors' list arguments doesn't provided !")
        else:
            dv.pieplot(x = args['x'] , y = args['y'] , colors = args['colors'])
        
    def Bar(self, args ,**argss):
        """
        Args here:
        args = { 'x': ('a' , 'b' , 'c', 'd' , 'f') , 
                 'y' = [1, 2, 3 , 4 , 5] 
                }
        """
        if ('x' and 'y' not in args.keys()) and type(args['x']) is not tuple and type(args['y']) is not list:
            raise TypeError("Excpected 'x' (touple) and 'y' (list) arguments doesn't provided !")
        else:
            dv.bar(x = args['x'] , y = args['y'] )#,  xlabel = args['xlabel'] if 'xlabel' in args.keys() , 
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

    def Read_json(self , path):
        ''' Read and return JSON file context from provided Path to the Json file!'''
        with codecs.open( path , 'r' , encoding="utf-8") as f:
                return json.load(f)

    def Read_text(self , path):
        ''' Read and return Text file context from provided Path to the Text file!'''
        with codecs.open( path , 'r' , encoding='utf-8') as f:
            return f.read()

    def Read_problems(self , dataset_root_dir , merge_candidates = False):
        """
        Reading problems candidates texts from txt files.
        Returning candidates text for each Problem in this Data Structors:
        Here merge_candidates variabel is 'False'
        all_candidates_txts = {
            'problem00001': { 'candidate00001': [txt1 , txt2 , txt3 , txt4, ...],
                             'candidate00002': [txt1 , txt2 , txt3 , txt4, ...],
                             .....
                             }
            'problem00002': { 'candidate00001': [txt1 , txt2 , txt3 , txt4, ...],
                             .....
                             }
            .....
            ..... }
        if merge_candidates variable is 'True'
        all_candidates_txts = { 'problem00001': { 'candidate00001': txt , 'candidate00002': txt, ...} 
                                'problem00002': { 'candidate00001': txt , .... } , .... }
        """
        all_candidates_txts = {}
        problems = self.Read_json(os.path.join( dataset_root_dir , "collection-info.json"))
        for problem in problems:
            print("working on problem : " , problem['problem-name'])
            problem_info = self.Read_json(os.path.join( dataset_root_dir , problem['problem-name'], 'problem-info.json'))
            candidates = [candidate['author-name'] for candidate in problem_info['candidate-authors']]
            candidates_txts = {}
            for candidate in candidates:
                txt = []
                for txt_name in os.listdir(os.path.join( dataset_root_dir , problem['problem-name'] , candidate)):
                    txt_path = os.path.join(os.path.join( dataset_root_dir , problem['problem-name'] , candidate , txt_name))
                    txt.append(self.Read_text(txt_path))
                if merge_candidates:
                    candidates_txts[candidate] = ' '.join(txt)
                else:
                    candidates_txts[candidate] = txt
            all_candidates_txts[problem['problem-name']] = candidates_txts
        return all_candidates_txts

class Style_Change_Detection(Pan):
    pass

class Celebrity_Profiling(Pan):
    pass

class Bots_And_Gender_Profiling(Pan):
    pass


def main():
    cross_domain_task = Cross_Domain_Authorship_Attribution()
    prediction_dic = cross_domain_task.Read_json_file("alternative_truth_json.json")
    prediction_normal = cross_domain_task.Normal_context(whole_documents = "on")
    cross_domain_task.Read_json_file("all_truth.json")
    cross_domain_task.Normal_context(whole_documents = "on")
    cross_domain_task.Confusion_matrix(prediction_normal)
    print('Accuracy is : ',cross_domain_task.Accuracy())
    print('Precision is : ', cross_domain_task.Precision())
    print('Recall is : ', cross_domain_task.Recall())
    print('F1 score is : ', cross_domain_task.F1())

#if __name__ == "__main__": main()
