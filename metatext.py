from typing import List,Tuple
import glob
import random
from functools import reduce
import csv
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
from neuthink.functional import usplit
import neuthink.functional as fp
import os

try:
    import sentencepiece as spm
    spm_enabled = True
except ModuleNotFoundError:
    spm_enabled = False


def str2dict(st: str)-> Dict[str, int]:
     d = {}
#     print(len(st))
#     print(st)
     for i,x in enumerate(st):
         d[x] = i
     return d


def list2dict(st:List[str])-> Dict[str,int]:
    d ={}
#     print(len(st))
#     print(st)
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
        self.alphabet_rus_char = 'ЙЦУКЕНГШЩЗХЪЭЖДЛОРПАВЫФЯЧСМИТЬБЮ'
#        self.alphabet_rus_char = 'ЙЦУКЕНГШЩЗХЪЭЖДЛОРПАВЫФЯЧСМИТЬБЮЁQWERTYUIOPLKJHGFDSAZXCVBNM'

        self.alphabet_rus_char = self.alphabet_rus_char.lower() + self.alphabet_rus_char
        self.alphabet_rus_char = self.alphabet_rus_char + '1234567890-=][()<>?!^*.,"%+/ \n'
        self.alphabet_rus_char = str2dict(self.alphabet_rus_char)

        self.alphabet_all_char = 'ЙЦУКЕНГШЩЗХЪЭЖДЛОРПАВЫФЯЧСМИТЬБЮЁQWERTYUIOPLKJHGFDSAZXCVBNM'

        self.alphabet_all_char = self.alphabet_all_char.lower() + self.alphabet_all_char
        self.alphabet_all_char = self.alphabet_all_char + '»«—1234567890-=][()<>?!^*.,:"%+/ \n'
        #self.alphabet_rus_char = self.alphabet_rus_char + '1234567890:;-=][()<>?!^*.,"%+/ \n'
        self.alphabet_all_char = str2dict(self.alphabet_all_char)

        self.alphabet_en_char = 'QWERTYUIOPLKJHGFDSAZXCVBNM"'
        self.alphabet_en_char = self.alphabet_en_char.lower() + self.alphabet_en_char
        self.alphabet_en_char = self.alphabet_en_char + "'1234567890-=!@$%^&*()[]{}_+?<>:;,. \t\n"

        self.alphabet_en_char = str2dict(self.alphabet_en_char)


    def VectorizeLookupChars(self, source='chunk',target='chunk_tensor', alphabet=None, filename=None, gen_classes_train=True, one_class_per_batch = False, random_seq_len = True, precision="normal"):
        def get_index(char:str, alphabet) -> int:
            if char in alphabet:
                return alphabet[char]
            else:
                return len(alphabet)

        if type(alphabet) is str:
         alphabet = self.model.resources['alphabet' + target]

        if alphabet is None and type(filename) is str:
            f = open(filename).readlines()
            alphabet = [x.split('\t')[0] for x in f]
            alphabet = list2dict(alphabet)
            self.model.resources['wordpieces'] = True
            self.model.resources['filename'] = filename
#        print(self.model.mode)
        if self.model.mode=='generate':
            return
        #print(alphabet)
        if self.model.mode=='design':
            if alphabet is None:
                alphabet = self.alphabet_rus_char
            self.alphabet = alphabet
#            print("inalpha",self.alphabet)
            print(len(self.alphabet))
            self.model.resources['alphabet' + target] = alphabet
            self.model.record("Text.VectorizeLookupChars",['source','target','alphabet', 'gen_classes_train', 'one_class_per_batch', 'random_seq_len'],[source,target,'alphabet' + target, gen_classes_train, one_class_per_batch, random_seq_len])

#            else:
#                self.alphabet = alphabet
#                self.model.resources['alphabet' + target] = alphabet
#                self.model.record("Text.VectorizeLookupChars",['source','target','alphabet'],[source,target,alphabet])
            self.model.last_call = target
            self.model.classes = [x for x in alphabet] + ['#']
            print("classes",len(self.model.classes))
#        print("Hello there!")
        if len(self.model)==0:
            return self
#        print("exec")
        start_i = 0

        if self.model.mode=='train':
            #if random_seq_len:
            #    start_i = random.randint(0,30)
            start_i = 0
        nodes = self.model[self.model.index: self.model.index + self.model.batch_size]
#        print("*********NM********")
#        print(self.model.batch_size)
#        print(self.model.index)
#        print(len(self.model))



        data_matrix = [[get_index(x, alphabet) for x in node[source][start_i:]] for node in nodes]
        #print(len(data_matrix))
        #print(len(data_matrix[1]))
#        print(data_matrix)
        tensor = torch.from_numpy(np.array(data_matrix))
        #print(tensor.size())
        if gen_classes_train and (self.model.mode == 'train' or self.model.mode == 'design' or self.model.mode=='eval'):
            if not one_class_per_batch:
                classes = tensor[: ,1:]
                tensor = tensor[:,:-1]
                self.model.metatensor["classes"] = classes

            if one_class_per_batch:
                classes = tensor[:, -1]
                tensor = tensor[:, :-1]
                self.model.metatensor["classes"] = classes

        self.model.metatensor[target] = tensor
        if self.model.device is not None:
            #migrate tensors to gpu

    #        print(self.model.device)
            self.model.metatensor[target] = self.model.metatensor[target].to(self.model.device)
            if "classes" in self.model.metatensor:
               self.model.metatensor["classes"] = self.model.metatensor["classes"].to(self.model.device)
        self.res = tensor
        if precision!="normal":
          self.model.metatensor[target] = self.model.metatensor[target].half()
        return self

    def VectorizeLookupTransformer(self, source='chunk',target='chunk_tensor', alphabet=None, filename=None, gen_classes_train = True, original_batch_size=None):
        def get_index(char:str, alphabet) -> int:
            if char in alphabet:
                return alphabet[char]
            else:
                return len(alphabet)

        if type(alphabet) is str:
         alphabet = self.model.resources['alphabet' + target]

        if alphabet is None and type(filename) is str:
            f = open(filename).readlines()
            alphabet = [x.split('\t')[0] for x in f]
            alphabet = list2dict(alphabet)
            self.model.resources['wordpieces'] = True
            self.model.resources['filename'] = filename

        if self.model.mode=='generate':
            return
        #print(alphabet)
        if self.model.mode=='design':
            if original_batch_size is None:
                original_batch_size = self.model.batch_size
            if alphabet is None:
                alphabet = self.alphabet_rus_char
            self.alphabet = alphabet
            print(len(self.alphabet))
            self.model.resources['alphabet' + target] = alphabet
            self.model.record("Text.VectorizeLookupTransformer",['source','target','alphabet', 'gen_classes_train', 'original_batch_size'],[source,target,'alphabet' + target, gen_classes_train, original_batch_size])
            self.model.last_call = target
            self.model.classes = [x for x in alphabet] + ['#']
            print("classes",len(self.model.classes))
        #print("Hello there!")
        if len(self.model)==0:
            return

        data_matrix = []

        chunk1 = self.model[self.model.index][source]
        chunk2 = self.model[self.model.index+1][source]
        chunk = chunk1 + chunk2
        # print(self.model.index, 'INDEX')
        for i in range(self.model.subindex,self.model.subindex + original_batch_size + 1):
            cut_chunk = chunk[i:i+len(chunk1)]
            data_matrix.append([get_index(x, alphabet) for x in cut_chunk])
            self.model.subindex = i
            if self.model.subindex == len(chunk1) - 1:
                self.model.subindex = 0
                self.model.batch_size = 1
            else:
                self.model.batch_size = 0
        # print(data_matrix)

        tensor = torch.from_numpy(np.array(data_matrix))
        if gen_classes_train and (self.model.mode == 'train' or self.model.mode == 'design' or self.model.mode=='eval'):
            classes = tensor[:, -1]
            tensor = tensor[:, :-1]
            self.model.metatensor["classes"] = classes

        self.model.metatensor[target] = tensor
        if self.model.device is not None:
            #migrate tensors to gpu

    #        print(self.model.device)
            self.model.metatensor[target] = self.model.metatensor[target].to(self.model.device)
            if "classes" in self.model.metatensor:
               self.model.metatensor["classes"] = self.model.metatensor["classes"].to(self.model.device)
        self.res = tensor
        return self

    def VectorizeLookupCharsCluster(self, source='chunk',target='chunk_tensor', alphabet=None, filename=None, gen_classes_train=True):
        def get_index(char:str, alphabet) -> int:
            if char in alphabet:
                return alphabet[char]
            else:
                return len(alphabet)

        if type(alphabet) is str:
         alphabet = self.model.resources['alphabet' + target]

        if self.model.mode=='generate':
            return
        #print(alphabet)
        if self.model.mode=='design':
            if alphabet is None:
                alphabet = self.alphabet_rus_char
            self.alphabet = alphabet
#            print("inalpha",self.alphabet)
            print(len(self.alphabet))
            self.model.resources['alphabet' + target] = alphabet
            self.model.record("Text.VectorizeLookupCharsCluster",['source','target','alphabet'],[source,target,'alphabet' + target])
            self.model.last_call = target
            self.model.classes = [x for x in alphabet] + ['#']
            print("classes",len(self.model.classes))
        #print("Hello there!")
        if len(self.model)==0:
            return

        if self.model.mode=='train':
           start_i = random.randint(0,30)
        else:
           start_i = 0
#        start_i=start_i+10
        nodes = [x for x in self.model[self.model.index].Children({'type':source})]
        if self.model.mode=='train':
            nodes = nodes[0:30]

        self.model.metatensor['cluster_id']= self.model[self.model.index]['cluster_id']

        data_matrix = [[get_index(x, alphabet) for x in node[source][start_i:]] for node in nodes]
        tensor = torch.from_numpy(np.array(data_matrix))
        if gen_classes_train and (self.model.mode == 'train' or self.model.mode == 'design' or self.model.mode=='eval'):
            classes = tensor[:,1:]
            tensor = tensor[:,:-1]
            self.model.metatensor["classes"] = classes

        self.model.metatensor[target] = tensor
        if self.model.device is not None:
            #migrate tensors to gpu
            self.model.metatensor[target] = self.model.metatensor[target].to(self.model.device)
            if "classes" in self.model.metatensor:
               self.model.metatensor["classes"] = self.model.metatensor["classes"].to(self.model.device)
#                self.model.metatensor["classes"] = self.model.metatensor["classes"].to('cuda:1')

        self.res = tensor
        return self


    def StringDecode(self,data, alphabet=None)->str:
        def decode_char(x:int, alphabet)-> str:
            for y in alphabet:
               if alphabet[y]==x:
                result = y.replace("▁", " ") if self.model.resources['input_mode'] == 'wordpieces' else y
                # print(result)
                return result
            return "#"
        if alphabet is None:
            alphabet = self.alphabet
        return "".join([decode_char(x, alphabet) for x in data])

    def StringEncode(self,data,alphabet=None):
        if alphabet is None:
            alphabet = self.alphabet
        if self.model.resources['input_mode'] == 'wordpieces':
            modelname = self.model.resources['filename'].split(".")[0]
            modelname = modelname + '.model'
            data = BPETokenize(data, modelname)
            data = data[0].split(' ')
        return [alphabet[x] if x in alphabet else len(alphabet) for x in data]


    def Word2Chunk(self, source="word",target="chunk"):
        def find_index(lst, nod):
#            print("MYLST:",len(lst))
            for i in range(len(lst)):
#                print(nod['id'],lst[i]['id'])
                if nod['id'] == lst[i]['id']:
                 # print(nod['id'])
                  return i
        if len(self.model)>0:
             #get all words
             curword:Node = self.model[self.model.index]
             sent:Node = curword.Parent({})
             #print(curword)
             words = [x[source] for x in sent.Children({"type":source})]
             max_words = len(self.model) - (self.model.index)
             if len(words)>max_words:
                words = words[:max_words]

             char_inds = []
             chunk = ""
             for i in range(len(words)):
                 chunk = chunk +  words[i] + " "
                 char_inds.append(len(chunk)-1)
             sent[target] = chunk
             #print(sent[target])
#             print(char_inds)
#             print(len(char_inds))
             sent[target+"_"+"inds_"] = char_inds
             #print(char_inds)
             #print(len(words))
             self.model.batch_size = len(words)
             self.model.Parent[0].index = find_index(self.model.Parent[0],sent)
        if self.model.mode=="design":
           self.model.record("Text.Word2Chunk",['source','target'],[source,target])
        return self


    def VectorizeLookup(self, source="word",target="word_tensor",embeddings=None, stem=False, batch_mode=False,max_len=20):
        if self.vecs is None:
           print("Loading word vectors...")
           self.vecs = wv.WordVector(embeddings)

       # print("vectorization call:",len(self.model))
        dynamic = False
        tensor = None
        if len(self.model)>0:
            if not batch_mode:
                curword:Node = self.model[self.model.index] if self.model.PointerNode is None else self.model.PointerNode
                sent:Node = curword.Parent({})
                words = [x[source] for x in sent.Children({"type":source})]
            #   print("xwords",words)
                max_words = len(self.model) - (self.model.index)
                if len(words)>max_words:
                    words = words[:max_words]
                if self.model.PointerNode is None:
                    self.model.batch_size = len(words)
                indices = [self.vecs.get_word_index(x) for x in words]
            else:
                batch_size = self.model.batch_size
                sents = self.model[self.model.index:self.model.index+batch_size]
                indices = np.zeros((len(sents),max_len),dtype=int)
                if  len(sents[0].Children({"type":source}))==0:
                            if self.model.mode=='design':
                               print("WARNING: data is not pre-tokenized, or has wrong format. Attempting to tokenize data dynamically")
                            dynamic = True

                for i,sent in enumerate(sents):
                    if dynamic:
                        words = usplit(sent[source],keep_new_line=False, return_positions=False)
                    else:
                        words = [x[source] for x in sent.Children({"type":source})]


                    indices_x = [self.vecs.get_word_index(x) for x in words]
                    for j,x in enumerate(indices_x):
                        if j<max_len:
                           indices[i,j] = indices_x[j]
                tensor =  torch.from_numpy(indices)




        else:
           sentence = {}
           indices = [1]

        if tensor is None:
            tensor = torch.from_numpy(np.array([indices]))

        self.model.metatensor[target] = tensor

        if self.model.device is not None:
            #migrate tensors to gpu
            self.model.metatensor[target] = self.model.metatensor[target].to(self.model.device)

        self.res = tensor
        if self.model.mode=="design":
           self.model.record("Text.VectorizeLookup",['source','target','embeddings','stem','batch_mode','max_len'],[source,target,embeddings,stem,batch_mode,max_len])
        self.model.last_call = target
        return self

    def GetEntities(self, wordname="word", classname="type"):
       en.entity_parser(self.model,classname,wordname)
       return self.model.parent_graph.Match({"type":"entity"})


    def Vectorize(self, source="word",target="word_tensor", embeddings = None):
         #determine language to be used
               text = self.model[0][source]
               if 'а' in text or 'о' in text or 'и' in text or 'е' in text or 'у' in text or 'я' in text:
                   lang = 'ru'
               else:
                   lang = 'en'
                   print("en")

               if embeddings is None:
                    if lang =='ru':
                       embeddings='ru_50'
                    else:
                       embeddings='en_100'

               if len(self.model[0][source].split())==1:
                   #for single word case

                    self.VectorizeWordSimple(source=source, target= target, embeddings=embeddings)
               else:
                   #for multi word text
                   print("multipart (vectorize+fold)")
                   self.VectorizeNbowSimple(source=source, target= target,embeddings=embeddings)
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

#            if self.model.mode=='predict':
               #nodes = self.model
               #print("pred",len(nodes))
            z = []
            for x in nodes:
                if stem:
                   stemmer = Stemmer.Stemmer('russian')
                   z.append(torch.from_numpy(self.vecs.project_text_nbow(stemmer.stemWord(x[source]))).float())
                else:
                   z.append(torch.from_numpy(self.vecs.project_text_nbow((x[source]))).float())


            res = [x for x in z]
            self.model.metatensor[target] = torch.stack(res,dim=0)
            if self.model.device is not None:
             #migrate tensors to gpu
             self.model.metatensor[target] = self.model.metatensor[target].to(self.model.device)



            if self.model.device is not None:
            #migrate tensors to gpu
              self.model.metatensor[target] = self.model.metatensor[target].to(self.model.device)


        if self.model.mode=="design":
           self.model.record("Text.VectorizeNbowSimple",['source','target','embeddings','stem'],[source,target,embeddings,stem])
        self.model.last_call = target
        return self

    def VectorizeStackSimple(self, source="word",target="word_tensor",embeddings=None, stem=False,maxlen=10):
        if self.vecs is None:
           print("Loading word vectors...")
           self.vecs = wv.WordVector(embeddings)

       # print("vectorization call:",len(self.model))
        if len(self.model)>0:

            nodes = self.model[self.model.index:self.model.index+self.model.batch_size]
          #  print(len(nodes))
          #  print("vectorization",self.model.index, self.model.batch_size,len(self.model))

#            if self.model.mode=='predict':
               #nodes = self.model
               #print("pred",len(nodes))
            z = []
              
            for x in nodes:
                if stem:
                   stemmer = Stemmer.Stemmer('russian')
                   z.append(torch.from_numpy(self.vecs.project_text_stack(stemmer.stemWord(x[source]), maxlen=maxlen)).float())
                else:
                   z.append(torch.from_numpy(self.vecs.project_text_stack((x[source]), maxlen=maxlen)).float())


            res = [x for x in z]
            self.model.metatensor[target] = torch.stack(res,dim=0)
            if self.model.device is not None:
             #migrate tensors to gpu
             self.model.metatensor[target] = self.model.metatensor[target].to(self.model.device)



            if self.model.device is not None:
            #migrate tensors to gpu
              self.model.metatensor[target] = self.model.metatensor[target].to(self.model.device)


        if self.model.mode=="design":
           self.model.record("Text.VectorizeStackSimple",['source','target','embeddings','stem', 'maxlen'],[source,target,embeddings,stem,maxlen])
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
    
    def MarkSpan(self, tagname, start_tag, end_tag, name):
       state = False 
       for i,x in enumerate(self.model):
        #  print(x[tagname],start_tag)
          if x[tagname]==start_tag:
             state=True
          if x[tagname]==end_tag:
             self.model[i][tagname] = name
             state=False
         
          if state:
           #  print(name)
             self.model[i][tagname] = name
    
    def Tokenize(self,source="sentence", wordname="word", make_sent_node=False):
        '''tokenizes text in specified source and returns unrolled result
        Args:
        source - where to get text
        wordname - name of the field where words will be stored
        make_sent_node - make a separate node for sentence or connect words directly
        
        '''
        for x in self.model:
            if make_sent_node:
                sent_node = Node(x.parent_graph,{"type":"sentence","name":source})
                x.Connect(sent_node)
            else:
                sent_node = x
            Tokenize(x[source],wordname = "word",sentence=sent_node)
        return self.model.Unroll()

    def Ngrams(self, size, wordname = "word"):
        d ={}
        for i in range(len(self.model)-3):
            ngram = self.model[i][wordname] + " " + self.model[i+1][wordname]
            d[ngram] = d.get(ngram,0) + 1
        return d

    def GetWordsInSpan(self,start,stop):
        l = []
        for x in self.model:

            if x["pos_start"]>=start and x["pos_end"]<=stop:
             l.append(x)
            # print(x["pos_end"], stop, x["word"])
            if x["pos_start"]==start and x["pos_end"]>stop:
           #  print(x["pos_end"], stop, x["word"])
             l.append(x)
        return l

    def AcronymsInText(self)->Dict[str,str]:
        """Makes acronyms dict from text
        
        Returns:
            Dict[str,str] -- acronyms dict
        """
        text = self.model.parent_graph.MatchOne({'type':'original_text'})['text']
        bracket = lambda y, z: [i for i in range(len(z)) if z[i] == y]
        open_bracket = bracket('(', text)
        close_bracket = bracket(')', text)
        acronym = [{'text':text[open_bracket[i]+1:close_bracket[i]], 'pos_start':open_bracket[i]+1, 'pos_end':close_bracket[i]} for i in range(len(open_bracket)) if text[open_bracket[i]+1].isupper() and text[open_bracket[i]+2].isupper() and '=' not in text[open_bracket[i]+1:close_bracket[i]]]
        return acronym

    def findAcronyms(self, fromModel = True)-> Dict[str, str]:
        '''makes dictionary of acronyms from entities using heuristics'''
        entities = self.model.parent_graph.Match({'type':'entity'})
        words = self.model.parent_graph.Match({'type':'word', 'word':"("})
        if fromModel:
            short_acronyms = entities.Match(lambda x: x['text'].isupper() and len(x['text'])<6)
        else:
            short_acronyms = self.AcronymsInText()
        acronyms_dict = {}
        for acronym in short_acronyms:
            bracket_func = lambda x, acronym: words.Match(lambda y: int(x['pos_end']) < int(y['pos_start']) < int(acronym['pos_start'])).NotEmpty()
            if fromModel:
                full_def = entities.Match(
                lambda x: int(x['pos_end'])==int(acronym['pos_start']-2) and x['entity_subtype']==acronym['entity_subtype'] and not(x['text'].isupper()) and bracket_func(x, acronym))
            else:
                full_def = entities.Match(
                lambda x: int(x['pos_end'])==int(acronym['pos_start']-2) and not(x['text'].isupper()) and bracket_func(x, acronym))
            if full_def.NotEmpty() and acronym['text'] not in acronyms_dict:
                acronyms_dict[acronym['text']] = full_def.First()['text']
        return acronyms_dict

    def resolveAcronymsDict(self, acronyms_dict:Dict[str, str]) -> None:
        '''performs acronym resolution across all enitites based on dictionary of acronyms'''
        entities = self.model.parent_graph.Match({'type':'entity'})
        for entity in entities:
            if entity['text'] in acronyms_dict:
                entity['full_text'] = acronyms_dict[entity['text']]
                original_node = entities.Match({'text': entity['full_text']})
                if original_node.NotEmpty():
                    entity.Connect(original_node[0], label='co-reference')


    def GetClosestEntity(self, concept:Node, entity_type:str):
        ''' Given entity X returns list of entities Y that a) match entity b) sorted by distance from concept'''

        sent_node = concept.Parent({})
        all_ents = sent_node.Children({'type':'entity','entity_subtype':entity_type})
        cur_pos = concept['pos_start']
        sorted_list = sorted(all_ents, key= lambda x: abs(x['pos_start']-cur_pos))
        return sorted_list


    def WordAnnotate(self,termset, classname, wordname='word', target='term', stem=True,lang='russian'):
        '''does greedy word-based annotation using termset'''
        stemmer = Stemmer.Stemmer(lang)
        for x in self.model:
            word =  x[wordname].lower() if not stem else stemmer.stemWord(x[wordname].lower())
            if word in termset:
                x[target] = classname
            else:
                x[target] = 'other'

def MakeBPEDict(filename:str, model_prefix:str, dict_size:int):
        """Make BPE dictionary from textfile

        Arguments:
            filename {str} -- path to textfile
            model_prefix {str} -- model name (example: model_prefix = 'test', output model name = test.model)
            dict_size {int} -- dictionary size
        """
        if spm_enabled:
            spm.SentencePieceTrainer.train('--input=' + filename +' --model_prefix=' + model_prefix+' --vocab_size='+ str(dict_size))
        else:
            print('Sentencepiece is not installed, please install from https://github.com/google/sentencepiece')

def BPETokenize(text:str, modelname:str, sp=None)->str:
    """Tokenize text with BPE dictionary

    Arguments:
        text {str} -- input text
        modelname {str} -- path to model, where we can get dictionary

    Returns:
        str -- text string, tokenized with BPE
    """
    if sp is None:
     sp = spm.SentencePieceProcessor()
     sp.load(modelname)
    enc = sp.EncodeAsPieces(text)
    enc = ' '.join(enc)
    return enc,sp


def LoadWords(filename, separator=" ", default_class="other"):
    import neuthink.metagraph as m
    graph = Graph()
    all_words :  m.dNodeList = m.dNodeList(graph)
    save_func = lambda nodelist, filename : SaveWords(nodelist, filename, separator = separator)
    all_words.save_func = save_func
    print(all_words.save_func)
    f = open(filename, "r")
    line_num = 0
    if separator!=" ":
      column_names = f.readline().strip().split(separator)
    else:
      column_names = f.readline().strip().split()

    for line in f:
        line = line.strip()
        line = line.replace("\ufeff","")
        line_num  = line_num  + 1
        word = Node(graph, {})
        if separator==" ":
          words = line.split()
#          print("split")
        else:
          words = line.split(separator)
        if len(words) == len(column_names):
            for i in range(0, len(column_names)):
                word[column_names[i]] = words[i]
            all_words.append(word)
        else:
            if default_class is None:
               print ("Error: on col " + str(line_num) + " not enough data and no default class provided, skipping")
            else:
            #fill specified columns
             for i in range(0, len(words)):
                word[column_names[i]] = words[i]
             for i in range(len(words), len(column_names)):
                word[column_names[i]] = default_class
             all_words.append(word)

    return all_words

                #less words then colums - fill with default class if given


def Tokenize(text:str, wordname='word', sentence=None, maxwords=1000,make_sentences=True, sentence_separator='.',keep_new_line=False):
   '''Tokenizes text into NodeList of words with their positions, also can make sentence layer (group words into sentences)'''

   import neuthink.metagraph as m
   graph = Graph()
   words :  m.dNodeList = m.dNodeList(graph)
   word_list = usplit(text,keep_new_line=keep_new_line)#text.replace('.',' . ').replace(',',' , ').replace('!',' ! ').replace(':', " : ").replace('"',' " ').replace('-', ' - ').replace('?',' ? ')
   original_text = Node(graph, {"type":"original_text", "text":text})
   #word_list = text.split()
   if sentence is None:
      sentence = Node(graph,{"type":"sentence"})
      original_text.Connect(sentence)

   for x in word_list:
         word = Node(sentence.parent_graph, {})
         word[wordname] = x[0]
         word["pos_start"] = x[1]
         word["pos_end"] = x[2]
         word['type'] = wordname
         words.append(word)
         sentence.Connect(word)
         if len(sentence.Children({}))>maxwords:
            sentence = Node(graph,{"type":"sentence"})
            original_text.Connect(sentence)
         if make_sentences:
             if word[wordname]==sentence_separator:
                 sentence = Node(graph,{"type":"sentence"})
                 original_text.Connect(sentence)
   return words

def SaveWords(nodelist, filename, separator=" "):
    fields = nodelist[0].dict.keys()
    f = open(filename , 'w', encoding='utf-8')
    f.write(separator.join([y for y in fields if (type(nodelist[0][y]) is str) and y!='id'])+'\n')
    for x in nodelist:
        f.write(separator.join([x[y] for y in fields if (type(x[y]) is str) and not (y=='id')]) +'\n')
    f.close()


def SaveColumn(sentences, filename:str, separator=" ", column_list=[],
                        ofs:int=None, lim:int=None, word_name='word') -> None:
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
                column_list = sentences.columns
            f = open(filename, 'w')

            f.write("  ".join(column_list) + "\n")
            for sentence in sentences[ofs:lim]:
                    words = sentence.Children({'type':word_name})
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


def LoadCSVCluster(filename:str, maxlines=None, cluster_size=30, input_mode = 'char'):
    import neuthink.metagraph as m
    graph = Graph()
    chunks :  m.dNodeList = m.dNodeList(graph)
    f = open(filename, encoding='utf-8')
    csvr = csv.reader(f,delimiter=',')
    names = next(csvr)
    new_cluster=False
    chunks.resources['input_mode'] = input_mode

    first_line = next(csvr)
    cluster_id = first_line[1]
    base_node = Node(graph,{'type':'text_cluster','cluster_id':cluster_id})
    base_list:List[Node] = []
    base_list.append(Node(graph,{'type':'chunk', 'chunk':first_line[0],'cluster_id':first_line[1]}))
    z = 0
    for x in csvr:
        new_cluster_id = x[1]
        if (new_cluster_id != cluster_id) or (len(base_list)==cluster_size):
        #if  (len(base_list)==cluster_size):

            for nod in base_list:
                base_node.Connect(nod)
            base_list=[]
            chunks.append(base_node)
            base_node = Node(graph,{'type':'text_cluster','cluster_id':new_cluster_id})
            cluster_id = new_cluster_id
            z = z + 1
            if maxlines is not None:
                if z>maxlines:
                    break
        base_list.append(Node(graph,{'type':'chunk', 'chunk':x[0],'cluster_id':new_cluster_id}))
    return chunks






def LoadText(filename:str,chunk_size:int = 300, input_mode = 'char', maxlines:int=0, nodeobject=None, fileobject=None):
    '''loads text from file, divided by evenly sized string chunks
    important: this function discards last chunk
    '''
    import neuthink.metagraph as m
    graph = Graph()
    chunks :  m.dNodeList = m.dNodeList(graph) if nodeobject is None else nodeobject
    chunks.clear()
    f = open(filename, encoding='utf-8') if fileobject is None else fileobject
    eof_reached = False
    #this is memory optimized read, that reads file into memory only once
    #so we have to use while loop
    chunks.resources['input_mode'] = input_mode
    if input_mode == 'char':
        i = 0
        while not eof_reached and (maxlines==0 or i<maxlines):
            buffer = f.read(chunk_size)
            if len(buffer)==chunk_size:
                chunks.append(Node(graph,{"chunk":buffer}))
                i = i + 1
            else:
                eof_reached=True
        return chunks
    else:
        z = 0
        buffer = []
        buf = ""
        string_buffer = ""
        while not eof_reached:
            print(z,end='\r')
            z = z + 1
            if z>maxlines and maxlines>0:
                chunks.loadfunc = lambda : LoadText(filename,chunk_size=chunk_size, input_mode=input_mode, maxlines=maxlines, nodeobject=chunks, fileobject=f)
                break
            string = f.read(chunk_size)
            string = string_buffer + string
            string_buffer = ""
            if len(string) < chunk_size:
                eof_reached = False
                break
            if "\n" in string:
                string = string.replace("\n", " ")

            try:
             ind = string[::-1].index(" ")
            except:
             ind = len(string)-1
            ind = string[::-1].index(" ")

            string_buffer = string[-ind:]
            string = string[:-ind]
            string_split = string.split(' ')
            buf = buf+" ".join(string_split)
            buf_split = buf.split(' ')
            if len(buf_split) >= chunk_size:
                for i in range(chunk_size):
                    buffer.append(buf_split[0])
                    buf_split.remove(buf_split[0])
                buf = " ".join(string_split)
                if len(buffer) == chunk_size:
                    chunks.append(Node(graph,{"chunk":buffer}))
                    buffer = []
                # elif len(buffer) < chunk_size and not eof_reached:
                #     chunks.append(Node(graph,{"chunk":buffer}))
                #     buffer = []
                else:
                    eof_reached=True
        return chunks

def LoadCSV(filename:str,separator:str=","):
    '''loads csv into nodelist using CSV reader'''
    import neuthink.metagraph as m
    print ("DEPRECATION WARNING: CSV reader in metatext is deprecated, use CSV reader in metastruct module")

    graph = Graph()
    lines :  m.dNodeList = m.dNodeList(graph)
    f = open(filename)
    csv_reader = csv.reader(f,delimiter=separator)
    first_column = True
    for row in csv_reader:
        if first_column:
            column_names = row
         #   print(column_names)
            first_column = False
        else:
            _node = {column_names[i]:row[i] for i in range(len(row))}
            node = Node(graph,_node)
            lines.append(node)
    return lines

def SaveCSV(data, filename:str,separator:str=",",column_list=None, writeheader=True):
    '''writes nodelist  to csv using CSV writer'''
    import neuthink.metagraph as m

    f = open(filename,'w')
    csv_writer = csv.writer(f,delimiter=separator)
    if writeheader:
        csv_writer.writerow(column_list)

    for row in data:
        data_list = [row[x] for x in column_list]
        csv_writer.writerow(data_list)
    f.close()


def LoadAnnFolder(folder_name: str, ann_ext: str = 'ann'):
    '''This function loads data from BRAT ann format, only terms are supported at the moment'''

    def parse_ann(filename: str):
        lines = [x.split('\t')[1].split(' ') for x in fp.load_lines(filename)]
        #print(lines)
        return lines

    import neuthink.metagraph as m
    names = glob.glob(folder_name+'/*.txt')
    ann_ext = '.' + ann_ext

    graph = Graph()
    total_list :  m.dNodeList = m.dNodeList(graph)

    for x in names:
        base_name = os.path.basename(x)
        core_name = base_name.split('.')[0]
        text = fp.load_file(x)
        ann = parse_ann(os.path.dirname(x)+'/'+core_name+ann_ext)
        tokens = Tokenize(text)
        for x in ann:
            try:
                start_term = int(x[1])
            except:
                continue
            if ';' in x[2]:
                x[2] =x[2].split(';')[1]
            end_term = int(x[2])
            term_class = (x[0])
            span_words = tokens.Text.GetWordsInSpan(start_term, end_term)
            for x in span_words:
                x['termtag'] = term_class

        z = tokens.ConvertTo()
        for x in z:
            children = x.Children({})
            basenode = Node(graph,x.dict)
            for y in children:
                zd = Node(graph,y.dict)
                basenode.Connect(zd)


            total_list.append(basenode)


    return total_list
        #Text.SaveColumn(tokens.ConvertTo(),core_name+'.conll', column_list=['word','termtag'])


def LoadColumn(filename: str, separator=" ",stop_word="<STOP>", maxlen=1000, strip_spaces=False, default_class='other', expand=False, maxwords=None, parent_graph = None):

        '''loads sentence view from file with columns file'''
        #TODO: needs simplification
        import neuthink.metagraph as m
        graph = Graph() if parent_graph == None else parent_graph
        sentences :  m.dNodeList = m.dNodeList(graph)

        print(sentences.save_func)
        f = open(filename, "r")
        column_names = split(f.readline().strip(), separator)
        cur_sentence = Node(graph, {})
        cur_sentence["words"] = NodeList(graph)
        cur_sentence['type']='sentence'
        #sentences.colums = column_names
        columns = column_names
        #save_func = lambda nodelist, filename : SaveColumn(nodelist, filename, separator = separator)
        save_func = None
        sentences.save_func = save_func
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
                return sentences
            if "<STOP>" in line or line == ""  or  len(cur_sentence["words"])>maxlen:
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
                        if 'pos_' in column_names[i]:
                            try:
                              word[column_names[i]] =  int(word[column_names[i]])
                            except:
                                 word[column_names[i]]=-128
                            #  print(column_names[i])
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
                            if 'pos_' in column_names[i]:
                             try:
                              word[column_names[i]] =  int(word[column_names[i]])
                             except:
                                 word[column_names[i]] = -128
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

def make_sent_from_node(node:Node[str, str], wordname="word")->str:
    '''this function takes sentence node with words as children and reconstructs text of the sentence, when no positions are avalible'''

    words = node.Children({'type':wordname})
    sent = " ".join([x[wordname] for x in words])
    sent = sent.replace(" ,",",").replace(" .",".").replace(" !","!").replace(" ?","?").replace("( ","(").replace(" )",")")
    return sent


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
