import re
import numpy as np
import codecs
import math
import os
import inspect
from typing import List
from functools import reduce

mpath = os.path.dirname(os.path.abspath(__file__))

def is_number(line:str)->bool:
  result = re.match("[\\d]{3,6}(\.[\d]0){0,1}",line)
  if result is None:
    return False
  else:
   return (len(result.group(0))==len(line))

def is_number2(line:str)->bool:
  result = re.match("[\\d]{1,10}(\.[\d]0){0,1}",line)
  if result is None:
    return False
  else:
   return (len(result.group(0))==len(line))

def make_dict(_list):
  dct = {}
  for px in _list:
    dct[px.strip()] = px.strip()
  return dct


latUp =  make_dict([x for x in 'QWERTYUIOPLKJJHGFDSAZXCVBNM'])
latDown = make_dict([x for x in 'QWERTYUIOPLKJJHGFDSAZXCVBNM'.lower()])
ruUp =    make_dict([x for x in 'ЙЦУКЕНГШЩЗХЪЭЖДЛОРПАВЫФЯЧСМИТЬБЮ'])
ruDown =  make_dict ([x for x in 'ЙЦУКЕНГШЩЗХЪЭЖДЛОРПАВЫФЯЧСМИТЬБЮ'.lower()])
digits =   make_dict ([x for x in '1234567890'])


def check_symbol (symb):
   symtype="o"
   if symb in latUp:
     symtype = "L"
   if symb in latDown:
     symtype = "l"
   if symb in ruUp:
     symtype = "R"
   if symb in ruDown:
     symtype = "r"
   if symb in digits:
     symtype = "D"
   return symtype

def genWordShape (line):
 wshape = ""
 psym = ""
 pindex = 0
 i = 0
 for i in range(0,len(line)):
   sym = (line[i])
   symtyp = check_symbol(sym)
   if psym != symtyp:
    wshape = wshape + symtyp
   psym = symtyp
 return ("##" + wshape)

def normalize(v:np.array)->np.array:
    norm=np.linalg.norm(v)
    if norm==0:
       return v
    return v/norm

def dist(v1,v2):
    '''computes Eulidean distance between two vectors'''
    return math.sqrt(sum((v1-v2)**2))

def dist_cosine(v1,v2):
    '''computes dot cosine distance'''
    return 0.0 - np.dot(normalize(v1),normalize(v2))


class WordVector():
    def __init__(self, filename:str)->None:
        '''Loads projections from file filename. example: ru_50
           reads from vectors_ru_50.txt and words_ru_50.txt'''
        self.projections=[]
        self.base_dictionary={}
        self.word_indexes=[]
        self.list_words = []
        self.load_projections("vectors_"+filename+'.txt','words_'+filename+'.txt')

    def save_projections(self,filename):
        vectors_name = "vectors_"+filename+'.txt'
        words_name = 'words_'+filename+'.txt'
        f_vec = open(vectors_name,'w')
        f_word = open(words_name,'w')

        for word in self.base_dictionary:
            p = ' '.join([str(x) for x in self.projections[self.base_dictionary[word]].tolist()])
            if len(self.projections[self.base_dictionary[word]])  == len(self.projections[0]):
                f_word.write(word+'\n')
                f_vec.write(p+'\n')
        f_vec.close()
        f_word.close()


    def load_projections(self, filename:str,filename1:str)->None:
        '''reading vectors from text files'''
        f = open(mpath + "/" + filename,"r")
        lines = f.readlines()
        #print(len(lines))
        f.close()
        for l in lines:
            l1 = l.split()
            vect = np.zeros(len(l1))
            for i in range(0,len(vect)):
                vect[i] = float(l1[i])
            self.projections.append(vect)
        #print(len(self.projections))
        #reading words to dictionary
        f = open(mpath + "/" + filename1,"r",encoding='utf-8')
        p = f.readlines()
        f.close()
        #print(len(p))
        for i in range(0,len(p)):
           word = p[i].lower()
           if not word in self.base_dictionary:
            self.base_dictionary[p[i].strip()] = i
            self.word_indexes.append(p[i].strip())
            self.list_words.append(p[i].strip())
        #print(len(self.base_dictionary))
        #print(len(self.word_indexes))

    def save_set_projections(self, filename1:str, filename2:str):
        print("saving set of pojections\n")
        f1 = open(mpath + "/" + filename1,"w")
        f2 = open(mpath + "/" + filename2,"w",encoding='utf-8')
        for x in self.base_dictionary.keys():
            #print(x)
            f2.write(x+"\n")
            pr = self.project_word(x)
            p = ""
            for i in pr:
                p+= str(i) + " "
            f1.write(p+"\n")
        f1.close()
        f2.close()
        self.projections.clear()
        self.word_indexes.clear()
        self.list_words.clear()

    def vocab_size(self):
        '''number of words loaded'''
        return len(self.list_words)
    def embedding_size(self):
        return len(self.projections[0])

    def projection_matrix(self):
        return (np.vstack(self.projections))

    def has_projection (self, word1:str)->bool:
        if (is_number2(word1)):
           word = "#number"
        else:
           word = word1.lower()

        return (word in self.base_dictionary)

    def get_word_index(self, word1:str)-> int:
        '''returns index of word in projections list'''
        if is_number2(word1):
            word = "#number"
        else:
            word = word1.lower()
        if word in self.base_dictionary:
            return(self.base_dictionary[word])
        else:
            shape =  (genWordShape(word)).lower()
            if shape in self.base_dictionary:
               return (self.base_dictionary[shape])
            else:
               return (self.base_dictionary["unk"])

    def project_word(self, word:str, ad_words=None)->np.array:
        '''returns projection of a given word'''
        word = word.lower()
        index = self.get_word_index(word)
        vector = self.projections[index]
        if len(vector)!=len(self.projections[0]):
            return self.projections[self.get_word_index('unk')]
            print(word)
        if (ad_words is None):
           return vector
        else:
          k = 0
          if word in ad_words:
             k = 1
          vector = np.hstack((vector, k))
        return vector


    def project_text_nbow(self, text:str)->np.array:
        ''' makes neural bag of word projection of text'''

        spl = text.lower().split()
        vector = np.zeros(len(self.projections[0]))
        cnt = 0
        for wrd in spl:
            if wrd in self.base_dictionary:
            # print(wrd)
             vect = self.project_word(wrd)
             vector = vector + vect
             cnt = cnt + 1
        if cnt == 0 :
           count = 1
        else:
           count = cnt
        return (vector) / count

        def project_text_nbow_fast(self, text:str)->np.array:
            ''' makes neural bag of word projection of text using np.sum'''

            spl = text.lower().split()
            vector = np.zeros(len(self.projections[0]))
            cnt = 0
            vect_list = [self.project_word(wrd) for wrd in spl if wrd in self.base_dictionary]
            if len(vect_list)>0:
                vector = np.sum(vect_list,axis=0)
            else:
                vector = np.zeros(len(self.projections[0]))
            return (vector) / max(1,len(vect_list))


    def project_text_stack(self, text:str,maxlen=10)->np.array:
        spl = text.split()
        beg = np.zeros(0)
        for wrd in spl[0:maxlen]:
            beg = np.append(beg,self.project_word(wrd))
        if len(spl)<maxlen:
         for i in range(maxlen-len(spl)):
            beg = np.append(beg,self.project_word("unk"))

        return beg

    def project_text_stack2d_pad(self, text:str, maxlen:int, ad_words = None):
        '''creates 2d matrix of stacked word vectors, padded to maxlen'''

        if type(text) == str:
           spl = text.split()
        else:
           spl = text
        lst = []
        for x in spl:

           lst.append(self.project_word(x, ad_words))
        while len(lst) < maxlen:
              lst.append(self.project_word("unk", ad_words))

        return lst[0:maxlen]

    def project_text_inv(self, text):
        spl = text.split().reverse()
        beg = np.zeros(0)
        for wrd in spl:
            beg = np.append(beg,self.project_word(wrd))
        return beg


    def find_closest_words(self, v1:np.array, nwords:int = 20, dist_func=dist)->List[str]:
        '''computes list of words (of length nwords) that are closest to a given word projection v1'''
        lst = []
        for word in self.list_words[0:100000]:
            wvec = self.project_word(word)
            dis = dist_func(v1,wvec)
            lst.append((dis,word))

        lst = sorted(lst)[0:20]
        lst1 = []
        for x in lst:
           lst1.append(x)

        return lst1

    def tree_finder(self, seed_word, lookup_depth=2, thresold=0.8):
        '''implements unsupervised dictionary construction for enitity tagging, using method of Blinov et al,2014 
        http://www.aclweb.org/anthology/S14-2020
        Args:
          seed_word - word from which to start
          lookup_depth - tree depth
          thresold - word similarity cutoff thresold
        
        '''
        candidates = self.find_closest_words(self.project_word(seed_word))
        candidates = [x[1] for x in candidates if x[0]>thresold]
        if lookup_depth==1:
            return candidates
        
        candidates = reduce(lambda x,y: x+y, [self.tree_finder(x,lookup_depth=lookup_depth-1) for x in candidates],[])
        return candidates




    def add_vector(self, word:str, vector:np.array)->None:
        ''' add new word vector to projections'''
        word = word.lower()
        self.projections.append(vector)
        index = len(self.projections)-1
        self.base_dictionary[word] = index





# print("loading word vectors")
# load_projections("vectors_rus.txt", "words_rus.txt")
# print(len(base_dictionary))
#
# vec = project_word("стол")
#
# add_vector("MatchOne", vec)
# k = base_dictionary["matchone"]
# print(base_dictionary["matchone"])
# print(projections[k])
#
# print("_______________________________________________")
#
# # vec = project_word("стул")
# #
# # add_vector("content", vec)
# # k = base_dictionary["content"]
# # print(base_dictionary["content"])
# # print(projections[k])
#
# print(project_word("MatchOne"))
