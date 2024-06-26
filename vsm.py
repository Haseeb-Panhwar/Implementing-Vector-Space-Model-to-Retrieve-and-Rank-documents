
import re,os
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import numpy as np
import json

class NumpyArrayEncoder(json.JSONEncoder): #Special Encoding for convering dictionary of numpy arrays to json 
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        return json.JSONEncoder.default(self, obj)


class vsm:  # main class
    ps = PorterStemmer()
    idf_vector = 0
    idf = {}
    stop_words = []
    docs=np.asarray([],dtype=np.int16)
    vsm_index = {}
    doc_tfidf = {}
    query_vector = 0
    output_filename = 'doc_tfidf.json'
    score = []
    re1=r'[a-z]+((?:-[a-z]+)*)?'

    def __init__(self) -> None:
        pass

    def prep(self):  # This function stems stopwords, re-stores them and creates numpy array of docs 
        with open("Stopword-List.txt","r") as f:
            self.stop_words = f.read().split()
        for i,w in enumerate(self.stop_words):
            self.stop_words[i] = ps.stem(self.stop_words[i])
        
        filenames=os.listdir('ResearchPapers/')
        for i in filenames:
            doc_id=int(i.strip('.txt'))
            self.docs = np.append(self.docs,(doc_id))
        self.docs = np.sort(self.docs)

    def isConsecChar(self,val): # This function calculates the number of times a character has occured in a string: Used in Preprocessing
        count = 0
        max_count=0
        for i,v in enumerate(val):
            if i==len(val)-1 and count>0:
                count+=1
                continue
            elif i<len(val)-1:
                if v==val[i+1]:
                    count+=1
                else:
                    count=0
            if count>max_count:
                max_count = count
        return max_count+1
    

    def build_temp_vsm_index(self):  # This function builds a dictionary of all terms of corpus and corresponding array of docs with their tfs
        n = len(self.docs)
        for i in self.docs:
            file = os.path.join('ResearchPapers',str(i)+'.txt')
            with open(file,'r',encoding='utf-8', errors='ignore') as f:
                text = f.read()
                filtered_text = self.filter_and_stem_text(text)
                for val in filtered_text:
                    if val in self.vsm_index:
                        loc = int((np.where(self.docs==i))[0]) # gets index of the current doc in a doc vector
                        self.vsm_index[val][loc]+=1
                    else:
                        temp_doc_vector = np.zeros(n)
                        loc = int((np.where(self.docs==i))[0])
                        temp_doc_vector[loc]+=1
                        self.vsm_index[val] = temp_doc_vector
    


    def build_idf(self): # Populates idf dictionary, term : idf value, for all terms in corpus
        self.idf_vector = np.zeros(len(self.vsm_index))
        n = len(self.docs)
        count=0
        for i,j in self.vsm_index.items():
            self.idf[i]=np.around(np.log10(n/sum(1 for x in j if x != 0)),3) # 1. Counts the non-zeros entries for each term in vsm_index, 2. calculates idf = log(N/df) and rounds it to 3 decimal places
            self.idf_vector[count] = self.idf[i]
            count +=1

    def filter_and_stem_text(self,text): # Main Text cleaning function
        stemmed_words = []
        for word in text.lower().split():
            if any(exclusion in word for exclusion in ["@", ".co", "http"]):
                continue
            if self.isConsecChar(word)>2:
                continue
            match = re.search(self.re1, word)
            if not match:
                continue
            word = match.group()
            if "-" in word and word.count('-') == 1:
                word = word.replace("-", "")
            stemmed_word = ps.stem(word)
            if stemmed_word not in self.stop_words and 1 < len(stemmed_word) < 16:
                stemmed_words.append(stemmed_word)
        return stemmed_words

    def build_tfidf(self): #Builds the tfidf index , doc id as keys : vector < axis for all terms with tf.idf values > 
        n=len(self.vsm_index)
        e_len = np.zeros(n)
        for doc in self.docs:
            file = os.path.join('ResearchPapers',str(doc)+'.txt')
            with open(file,'r',encoding='utf-8', errors='ignore') as f:
                text = f.read()
                filtered_text = self.filter_and_stem_text(text) # filters / cleans data
                for idx, term in enumerate(self.vsm_index):
                    term_frequency = filtered_text.count(term)
                    if doc not in self.doc_tfidf:
                        self.doc_tfidf[doc] = np.zeros(len(self.vsm_index))
                    self.doc_tfidf[doc][idx] = np.around(term_frequency * self.idf[term], 3) # store tf.idf for each term

            e_len = np.sqrt(np.sum(np.square(self.doc_tfidf[doc]))) # e_len is the euclidian distance for each doc
            self.doc_tfidf[doc] = np.around(self.doc_tfidf[doc]/e_len,3) # divided e_len with tf.idf vector
            self.doc_tfidf[doc][np.isnan(self.doc_tfidf[doc])]=0 # check if nan values equates them to zero

    
    def store_tfidf(self): # Stores tf.idf in json file
        doc_tfidf_str_keys = {str(key): value.tolist() if isinstance(value, np.ndarray) else value for key, value in self.doc_tfidf.items()}
        # Save the dictionary to a JSON file
        with open(self.output_filename, 'w') as json_file:
            json.dump(doc_tfidf_str_keys, json_file, cls=NumpyArrayEncoder)
    
    def loadtfidf(self): # loads tf.idf dictionary from json file
        # Load the dictionary from a JSON file
        with open(self.output_filename, 'r') as json_file:
            doc_tfidf_loaded = json.load(json_file)
        # Convert keys back to integers and lists back to numpy arrays if necessary
        self.doc_tfidf = {int(key): np.array(value) for key, value in doc_tfidf_loaded.items()}

    def getquery(self,query): # appliedprocessing on query as a document
        print(len(self.vsm_index))
        self.query_vector = np.zeros(len(self.vsm_index))
        count=0
        stemmed_query = self.filter_and_stem_text(query)
        print(stemmed_query)
        for term in self.vsm_index:
            if term in stemmed_query:
                self.query_vector[count] = stemmed_query.count(term)
            count+=1
        self.query_vector = self.query_vector*self.idf_vector
        e_len = np.sqrt(np.sum(np.square(self.query_vector)))
        self.query_vector = np.around(self.query_vector/e_len,3)
        self.query_vector[np.isnan(self.query_vector)]=0

    def getscore(self,alpha): 
        self.score=[]
        for doc in self.docs:
            self.score.append((doc,np.around(np.sum(self.doc_tfidf[doc]*self.query_vector),3)))
        self.score = sorted(self.score,key=lambda x: x[1],reverse=True)
        self.score = [x for x in self.score if x[1]>alpha]


# v = vsm()
# v.prep()
# v.build_temp_vsm_index()
# v.build_idf()
# print(len(v.idf))
# v.build_tfidf()
# v.store_tfidf()
# v.loadtfidf()

# v.loadtfidf()
# print(v.doc_tfidf)

# v.getquery()
# print(len(v.vsm_index))
# print(v.query_vector)
# v.getscore()





        


