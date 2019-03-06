from typing import List,Tuple
import glob
import random
from functools import reduce

import numpy as np
from neuthink.nImage  import mvImage
from neuthink.graph.basics import Graph,NodeList,Node
from neuthink.nftools import split
import torch
import torch.nn as nn
import Stemmer
from neuthink.wordvectors import word_vector as wv
import neuthink.nlptools.entities as en
from typing import Dict

def str2dict(st:str)-> Dict[str,int]:
     d ={}
     for i,x in enumerate(st):
         d[x] = i
     return d

class MetaText(object):
    """Image class, as part of metamodel"""
    def __init__(self, ParentModel):
        super(MetaText, self).__init__()
        self.model = ParentModel
        self.graph = ParentModel.parent_graph
        self.vecs = None
        self.alphabet = None
#        self.alphabet_rus_char = 'ЙЦУКЕНГШЩЗХЪЭЖДЛОРПАВЫФЯЧСМИТЬБЮ'
        self.alphabet_rus_char = 'ЙЦУКЕНГШЩЗХЪЭЖДЛОРПАВЫФЯЧСМИТЬБЮЁQWERTYUIOPLKJHGFDSAZXCVBNM'

        self.alphabet_rus_char = self.alphabet_rus_char.lower() + self.alphabet_rus_char
        self.alphabet_rus_char = self.alphabet_rus_char + '»«—1234567890-=][()<>?!^*.,:"%+/ \n'
        #self.alphabet_rus_char = self.alphabet_rus_char + '1234567890:;-=][()<>?!^*.,"%+/ \n'
        self.alphabet_rus_char = str2dict(self.alphabet_rus_char)

        
    
    
    def VectorizeLookupChars(self, source='chunk',target='chunk_tensor', alphabet=None, gen_classes_train=True):
        def get_index(char:str, alphabet) -> int:
            if char in alphabet:
                return alphabet[char]
            else:
                return len(alphabet)
        
        if type(alphabet) is str:
         alphabet = self.model.resources['alphabet' + target]

        

        if self.model.mode=='generate':
        #    print("return/gen")
            return
        #print(alphabet)
        if self.model.mode=='design':
            if alphabet is None:
                alphabet = self.alphabet_rus_char
                self.alphabet = alphabet
                print(len(self.alphabet))
                self.model.resources['alphabet' + target] = alphabet
                self.model.record("Text.VectorizeLookupChars",['source','target','alphabet'],[source,target,'alphabet' + target])

            else:
                self.model.record("Text.VectorizeLookupChars",['source','target','alphabet'],[source,target,alphabet])
            self.model.last_call = target
            self.model.classes = [x for x in alphabet] + ['#']
            print(len(self.model.classes))
        #print("Hello there!")
        if len(self.model)==0:
            return

        if self.model.mode=='train':
           start_i = random.randint(0,30)
        else:
           start_i = 0
        nodes = self.model[self.model.index: self.model.index + self.model.batch_size]
        data_matrix = [[get_index(x, alphabet) for x in node[source][start_i:]] for node in nodes]
        #print(len(data_matrix))
        #print(len(data_matrix[1]))

        tensor = torch.from_numpy(np.array(data_matrix))
        #print(tensor.size())
        if gen_classes_train and (self.model.mode == 'train' or self.model.mode == 'design'):
            classes = tensor[:,1:]
            tensor = tensor[:,:-1]
            self.model.metatensor["classes"] = classes
        #move to GPU!
        self.model.metatensor[target] = tensor

        if self.model.device is not None:
#           print("toGPU")

           self.model.metatensor[target] = self.model.metatensor[target].to(self.model.device)
           if "classes" in self.model.metatensor:
              self.model.metatensor["classes"] = self.model.metatensor["classes"].to(self.model.device)
              


        self.res = tensor
        return self


    def StringDecode(self,data, alphabet=None)->str:
        def decode_char(x:int, alphabet)-> str:
            for y in alphabet:
               if alphabet[y]==x:
                 return y
            return "#"
        if alphabet is None:
            alphabet = self.alphabet
        return "".join([decode_char(x, alphabet) for x in data])

    def StringEncode(self,data,alphabet=None):
        if alphabet is None:
            alphabet = self.alphabet
        return [alphabet[x] if x in alphabet else len(alphabet)+1 for x in data]

    def Word2Chunk(self, source="word",target="chunk"):
        def find_index(lst, nod):
            for i in range(len(lst)):
                if nod['id'] == lst[i]['id']:
                  return i
        if len(self.model)>0:
             #get all words
             curword:Node = self.model[self.model.index]
             sent:Node = curword.Parent({})
             words = [x[source] for x in sent.Children({})]
             max_words = len(self.model) - (self.model.index)
             if len(words)>max_words:
                words = words[:max_words]
             char_inds = []
             chunk = ""
             for i in range(len(words)):
                 chunk = chunk +  words[i] + " "
                 char_inds.append(len(chunk)-1)
             sent[target] = chunk
             sent[target+"_"+"inds_"] = char_inds
             self.model.batch_size = len(words)
             self.model.Parent[0].index = find_index(self.model.Parent[0],sent)
        if self.model.mode=="design":
           self.model.record("Text.Word2Chunk",['source','target'],[source,target])     
        return self


    def VectorizeLookup(self, source="word",target="word_tensor",embeddings=None, stem=False):
        if self.vecs is None:
           print("Loading word vectors...")
           self.vecs = wv.WordVector(embeddings)

       # print("vectorization call:",len(self.model))
        if len(self.model)>0:
            curword:Node = self.model[self.model.index]
            sent:Node = curword.Parent({})
            words = [x[source] for x in sent.Children({})]
         #   print("xwords",words)
            max_words = len(self.model) - (self.model.index)
            if len(words)>max_words:
                words = words[:max_words]

            self.model.batch_size = len(words)
            indices = [self.vecs.get_word_index(x) for x in words]
        else:
           sentence = {}
           indices = [1]

        tensor = torch.from_numpy(np.array([indices]))
        if self.model.device is not None:
          tensor = tensor.to(self.model.device)
        self.model.metatensor[target] = tensor
        self.res = tensor
        if self.model.mode=="design":
           self.model.record("Text.VectorizeLookup",['source','target','embeddings','stem'],[source,target,embeddings,stem])
        self.model.last_call = target
        return self

    def GetEntities(self, wordname="word", classname="type"):
       en.entity_parser(self.model,classname,wordname)
       return self.model.parent_graph.Match({"type":"entity"})


    def Vectorize(self, source="word",target="word_tensor"):
         #determine language to be used
               text = self.model[0][source]
               if 'а' in text or 'о' in text or 'и' in text or 'е' in text or 'у' in text or 'я' in text:
                   lang = 'ru'
               else:
                   lang = 'en'

               if len(self.model[0][source].split())==1:
                   #for single word case
                   if lang =='ru':
                      self.VectorizeWordSimple(source=source, target= target, embeddings='ru_50c')
                   else:
                      self.VectorizeWordSimple(source=source, target= target, embeddings='en_100')
               else:
                   #for multi word text
                   print("multipart (vectorize+fold)")
                   if lang =='ru':
                      self.VectorizeNbowSimple(source=source, target= target,embeddings='ru_50c')
                   else:
                      print("en")
                      self.VectorizeNbowSimple(source=source, target= target,embeddings='en_100')
               return self

    def VectorizeNbowSimple(self, source="word",target="word_tensor",embeddings=None, stem=False):
        if self.vecs is None:
           print("Loading word vectors...")
           self.vecs = wv.WordVector(embeddings)

       # print("vectorization call:",len(self.model))
        if len(self.model)>0:

            nodes = self.model[self.model.index:self.model.index+self.model.batch_size]
          #  print(len(nodes))
          #  print("vectorization",self.model.index, self.model.batch_size,len(self.model))

            if self.model.mode=='predict':
               nodes = self.model
               print("pred",len(nodes))

            for x in nodes:
                if stem:
                   stemmer = Stemmer.Stemmer('russian')
                   x[target] = torch.from_numpy(self.vecs.project_text_nbow(stemmer.stemWord(x[source]))).float()
                else:
                   x[target] = torch.from_numpy(self.vecs.project_text_nbow((x[source]))).float()

            res = [x[target] for x in nodes]
            self.model.metatensor[target] = torch.stack(res,dim=0)

        if self.model.mode=="design":
           self.model.record("Text.VectorizeNbowSimple",['source','target','embeddings','stem'],[source,target,embeddings,stem])
        self.model.last_call = target
        return self



    def VectorizeWordSimple(self, source="word",target="word_tensor",embeddings=None, stem=False):

        if self.vecs is None:
           print("Loading word vectors...")
           self.vecs = wv.WordVector(embeddings)

       # print("vectorization call:",len(self.model))
        if len(self.model)>0:

            nodes = self.model[self.model.index:self.model.index+self.model.batch_size]
          #  print(len(nodes))
          #  print("vectorization",self.model.index, self.model.batch_size,len(self.model))

            if self.model.mode=='predict':
               nodes = self.model
               print("pred",len(nodes))

            for x in nodes:
                if stem:
                   stemmer = Stemmer.Stemmer('russian')
                   x[target] = torch.from_numpy(self.vecs.project_word(stemmer.stemWord(x[source]))).float()
                else:
                   x[target] = torch.from_numpy(self.vecs.project_word((x[source]))).float()

            res = [x[target] for x in nodes]
            self.model.metatensor[target] = torch.stack(res,dim=0)

        if self.model.mode=="design":
           self.model.record("Text.VectorizeWordSimple",['source','target','embeddings','stem'],[source,target,embeddings,stem])
        self.model.last_call = target
        return self

    def Tokenize(self,source="sentence", wordname="word"):
        '''tokenizes text in specified source and returns unrolled result'''
        for x in self.model:
            Tokenize(x[source],wordname = "word",sentence=x)
        return self.model.Unroll()

    def Ngrams(self, size, wordname = "word"):
        d ={}
        for i in range(len(self.model)-3):
            ngram = self.model[i][wordname] + " " + self.model[i+1][wordname]
            d[ngram] = d.get(ngram,0) + 1
        return d





    def WordAnnotate(self,termset, classname, wordname='word', target='term', stem=True,lang='russian'):
        '''does greedy word-based annotation using termset'''
        stemmer = Stemmer.Stemmer(lang)
        for x in self.model:
            word =  x[wordname].lower() if not stem else stemmer.stemWord(x[wordname].lower())
            if word in termset:
                x[target] = classname
            else:
                x[target] = 'other'







def LoadWords(filename, separator=" ", default_class="other"):
    import neuthink.metagraph as m
    graph = Graph()
    all_words :  m.dNodeList = m.dNodeList(graph)
    f = open(filename, "r")
    line_num = 0
    column_names = f.readline().strip().split(separator)
    for line in f:
        line = line.strip()
        line = line.replace("\ufeff","")
        line_num  = line_num  + 1
        word = Node(graph, {})
        words = line.split(separator)
        if len(words) == len(column_names):
            for i in range(0, len(column_names)):
                if i>=len(column_names):
                   print(words[i],"this")
                if i>len(words):
                   print(words[i],"that")
                word[column_names[i]] = words[i]
            all_words.append(word)
        else:
            if len(words)>len(column_names):
              continue
            if default_class is None:
               print ("Error: on col " + str(line_num) + " not enough data and no default class provided, skipping")
            else:
            #fill specified columns
             print(words)
             print(len(words))
             for i in range(0, len(words)):
                word[column_names[i]] = words[i]
             for i in range(len(words), len(column_names)):
                word[column_names[i]] = default_class
             all_words.append(word)

    return all_words

                #less words then colums - fill with default class if given


def Tokenize(text:str, wordname='word', sentence=None, maxwords=1000):
   import neuthink.metagraph as m
   graph = Graph()
   words :  m.dNodeList = m.dNodeList(graph)
   text = text.replace('.',' . ').replace(',',' , ').replace('!',' ! ').replace(':', " : ").replace('"',' " ').replace('-',  '-').replace('?',' ? ')
   word_list = text.split()
   if sentence is None:
      sentence = Node(graph,{"type":"sentence"})

   for x in word_list:
         word = Node(sentence.parent_graph, {})
         word[wordname] = x
         word['type'] = wordname
         words.append(word)
         sentence.Connect(word)
         if len(sentence.Children({}))>maxwords:
            sentence = Node(graph,{"type":"sentence"})
   return words

def SaveWords(nodelist, filename, separator=" "):
    fields = nodelist[0].dict.keys()
    f = open(filename , 'w', encoding='utf-8')
    f.write(separator.join(fields)+'\n')
    for x in nodelist:
        f.write(separator.join([x[y] for y in x.dict.keys() if (type(x[y]) is str) and not (y=='id') and not (y=='word')]) +'\n')
    f.close()


def SaveColumn(self, filename:str, separator=" ", column_list=[],
                        ofs:int=None, lim:int=None) -> None:
            '''saves view data to column file

            Args:
              column_list: list of colums to include in file to save,
               default is all colums
               lim: limit of sentences to save
               ofs: offset sentences to save'''
            #ISSUE: offset and limit will be removed from this function
            #Reason: recommended way to do this is to create another view, make
            #subset of data and then save it
            if ofs is not None or lim is not None:
                print ("ofs and lim parameters are deprecated, please update your code")

            if len(column_list) == 0:
                column_list = self.columns
            f = open(filename, 'w')

            f.write("  ".join(column_list) + "\n")
            for sentence in self[ofs:lim]:
                    words = sentence.Children({})
                    for word in words:
                        save_str = ""
                        #if word["word"].strip() != "":
                        for cl in column_list:
                                if cl in word:
                                    save_str = save_str + str(word[cl]) + "  "
                                else:
                                    save_str = save_str + "other" + "  "
                        f.write(save_str + "\n")
                    f.write("<STOP> \n")


def LoadText(filename:str,chuck_size:int = 300):
    '''loads text from file, divided by evenly sized string chunks
    important: this function discards last chunk
    '''
    import neuthink.metagraph as m
    graph = Graph()
    chunks :  m.dNodeList = m.dNodeList(graph)
    f = open(filename, encoding='utf-8',errors='ignore')
    eof_reached = False
    #this is memory optimized read, that reads file into memory only once
    #so we have to use while loop
    while not eof_reached:
        buffer = f.read(chuck_size)
        if len(buffer)==chuck_size:
            chunks.append(Node(graph,{"chunk":buffer}))
        else:
            eof_reached=True
    return chunks




def LoadColumn(filename: str, separator=" ", maxlen=1000, strip_spaces=False, default_class='other', expand=False, maxwords=None):

        '''loads sentence view from file with columns file'''
        #TODO: needs simplification
        import neuthink.metagraph as m
        graph = Graph()
        sentences :  m.dNodeList = m.dNodeList(graph)
        save_func = lambda nodelist, filename : SaveWords(nodelist, filename, separator = separator)
        sentences.save_func = save_func
        print(sentences.save_func)
        f = open(filename, "r")
        column_names = split(f.readline().strip(), separator)
        cur_sentence = Node(graph, {})
        cur_sentence["words"] = NodeList(graph)
        #sentences.colums = column_names
        columns = column_names
        line_num = 0
        #sentences.extended = expand
        for line in f:
            if strip_spaces:
               line = line.strip()
               line = line.replace("\ufeff","")
            else:
                line = line.replace('\n','')

            line_num  = line_num  + 1
            if (maxwords is not None) and (line_num>maxwords):
                return
            if "<STOP>" in line or line == "" or len(cur_sentence["words"])>maxlen:
                if len(cur_sentence["words"]) > 0:
                    sentences.append(cur_sentence)
                    cur_sentence = Node(graph, {})
                    cur_sentence["words"] = NodeList(graph)
                    cur_sentence["type"] = "sentence"
            else:
                word = Node(graph, {})
                words = split(line, separator)
        #        print (len(words))
         #       print (len(column_names))
                #TODO: This code is horrible: need to simplify
                #normal situation
                if len(words) == len(column_names):
                    for i in range(0, len(column_names)):
                        word[column_names[i]] = words[i]
                        if expand == True:
                            nn = Node(
                                graph,{"type":column_names[i],"value":words[i]})
                            word.Connect(nn)
                #less words then colums - fill with default class if given
                if len(words) < len(column_names):

                    if default_class is None:
                        print ("Error: on col " + str(line_num) + " not enough data and no default class provided, skipping")
                    else:
                        #fill specified columns
                        for i in range(0, len(words)):
                            word[column_names[i]] = words[i]
                            if expand is True:
                               nn=  Node(
                                    graph,{"type":column_names[i], "value":words[i]})
                               word.Connect(nn)
                        #fill unspecified colums with default value
                    #    print (range(len(words), len(self.columns)))
                     #   print (words)
                        for i in range(len(words), len(columns)):
                            word[column_names[i]] = default_class
                            if expand is True:
                                nn = Node(
                                    graph,{"type":column_names[i], "value":default_class})
                                word.Connect(nn)

                if len(words) > len(column_names):
                     #we have too many classes
                     #fill normal range
                     for i in range(0, len(column_names)):
                         word[column_names[i]] = words[i]
                         if expand == True:
                             nn = Node(
                                 graph,{"type":column_names[i], "value":words[i]})
                             word.Connect(nn)
                     #if expand = true, fill load other classes into graph

                     for i in range(len(column_names) , len(words)):
                          if expand == True:
                            nn = Node(
                                graph,{"type":column_names[-1],"value":words[i]})
                            word.Connect(nn)
                   #  print (word.children({"type":column_names[-1]}).Distinct("value"))

                word["type"] = "word"

                cur_sentence["words"].append(word)
                cur_sentence.Connect(word)
        if len(cur_sentence["words"]) > 0:
            sentences.append(cur_sentence)
        return sentences

#small auxiliary functions

def stem_list(wordlist, lang='russian'):
    stemmer = Stemmer.Stemmer(lang)
    return [stemmer.stemWord(x) for x in wordlist]

def contains_term(termset, text, stem=True, lang='russian'):
    '''checks if text contains any term from termlist'''
    tokens = Tokenize(text.lower(),wordname="word")
    if stem:
     stemmer = Stemmer.Stemmer(lang)
     for x in tokens:
         x["word"] = stemmer.stemWord(x["word"])
    c = False
    for x in tokens:
        if x["word"] in termset:
            c = True
    return c
