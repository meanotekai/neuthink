from typing import List
from neuthink.graph.basics import Node
from neuthink.nImage import  mvImage
import neuthink.metaimage as mimage
import neuthink.metatext
import torch
import torch.nn as nn
import numpy as np
from functools import reduce
from neuthink.display import display_comp_status
import types
import os
import random
import json
import sys
from shutil import copy as copyfile
import math
from sru import SRU


#we want all runs to be replicable
torch.manual_seed(65536)
random.seed(65536)

def unisize(q:torch.tensor):
    s = q.size()
    s = s[:-1]
    return reduce(lambda x,y:x*y , s)

def full_size(q:torch.tensor):
    s = q.size()
    return reduce(lambda x,y:x*y , s)


class TrackedNodeList(list):
    def __init__(self, parent_graph, *args):
        list.__init__(self, *args)
        self.parent_graph = parent_graph

    def __matchnode(self,node1, node2):
        is_match = True
        for x in node1:
            if x in node2:
                if node1[x] == node2[x]:
                    pass
                else:
                    is_match = False
            else:
                is_match = False
        return is_match

    def __getitem__(self, given):
        if isinstance(given, slice):
            # do your handling for a slice object:
            return type(self)(self.parent_graph, list.__getitem__(self, given))
        else:
            # Do your handling for a plain index
            return list.__getitem__(self, given)

    def First(self):
        if len(self)>0:
            return self[0]
        else:
            return self.parent_graph.Empty()

    def Delete(self):
       for x in self:
           x.Delete()

       return type(self)(self.parent_graph)

    def Set(self,field,value):
       for x in self:
           x.Set(field,value)

       return self

    def Detach(self,node):
       for x in self:
           x.Detach(node)

       return self

    def children_all(self, cond={}):
        '''Return all nodes that are connected to this node and
        satisfying condition <cond> that can be dictionary or bool function'''
        if len(self) == 1:
            all_children = self[-1].ConnectsTo()
            return all_children.Match(cond)
        else:
            res = self.parent_graph.Match(cond)
            return res

    def AddParent(self,node):
       for x in self:
           x.AddParent(node)

       return self

    def Mapi(self, func,source,target):

        for i,x in enumerate(self):
            x[target] = func(x[source],i,self)

    def Map(self, func, source, target):
        for x in self:
             x[target] = func(x[source])

    def HasChild(self, node):
        result = []
        for x in self:
            people = x.children(node)
            if people:
                result.append(x)
        return result

    def LeafNodes(self):
        '''returns nodes that have no children'''
        leaf_list = type(self)(self.parent_graph)
        for x in self:
            if len(x.Children({}==0)):
                leaf_list.append(x)
        return leaf_list

    def Distinct(self, node_property):
        '''Returns list of distinct values of specified node property'''
        d = {}
        for x in self:
            if node_property in x:
                d[x[node_property]] = 1
        result = [x for x in d if x != '']
        return result

    def MatchOne(self,node):
        return self.Match(node).First()

    def NotMatch(self,node):

        if type(node) is dict:
            match_list = type(self)(self.parent_graph)
            for x in self:
                match = self.__matchnode(node, x)
                if not match:
                    match_list.append(x)
        return match_list

    def Match(self,node):

        if type(node) is dict or type(node) is Node:
            match_list = type(self)(self.parent_graph)
            for x in self:
                match = self.__matchnode(node, x)
                if match:
                    match_list.append(x)

        if type(node) is types.FunctionType:
            match_list = type(self)(self.parent_graph)
            for x in self:
                match = node(x)
                if match:
                    match_list.append(x)


        return match_list

    def NotEmpty(self):
        return len(self) > 0

    def CountBy(self, source):
        d = {}
        for x in self:
            d[x[source]] = d.get(x[source],0) + 1
        return d



    def Empty(self):
        return len(self) == 0

    def Count(self):
        return len(self)

    def ConnectsTo(self):
        '''Return all nodes that are connected to this node'''
        return self.parent_graph.ConnectsTo(self)
    def __add__(self,new):
        _list = type(self)(self.parent_graph)
        for x in self:
            _list.append(x)
        for x in new:
            _list.append(x)
        return _list

def tin_size(t:torch.tensor) -> int:
    '''returns size of the tensor, except batch dimension'''
    return reduce(lambda x,y: x*y, list(t.size())[1:])

def dim_size(t:torch.tensor) -> int:
    return t.size()[2]

class dNodeList(TrackedNodeList,torch.nn.Module):
    def __init__(self, parent_graph, *args):
        TrackedNodeList.__init__(self, parent_graph, *args)
        torch.nn.Module.__init__(self)
        self.parent_graph = parent_graph
        self.Image = mimage.MetaImage(self)
        self.Text = neuthink.metatext.MetaText(self)
        self.mode = "design"
        self.code:str = ""
        self.metatensor={}
        self.res = None
        self.batch_size = 50
        self.index = 0
        self.naming = {}
        self.save_func = None
        self.class_source = None
        self.layers = {}
        self.loss = 0
        self.classes=[]
        self.test_set= None
        self.features={}
        self.func = None
        self.resources={}
        self.Parent = None
        self.Child = None
        self.device=None

    def  __hash__(self):
        '''for pytorch 1.0 compatibility. not very good idea, but not likely to cause any trouble'''
        return super(torch.nn.Module,self).__hash__()
    
    def enableGPU(self,device='cuda:1'):
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.device = device
        self.to(device)
    def disableGPU(self):
        self.to(torch.device("cpu"))
        self.device=None
             
    #auxiliary functions
    def vectorize_classes(self, nodes, class_name)->torch.tensor:
          '''makes a list of class numbers from nodelist nodes'''

          xl = [self.classes.index(x[class_name]) for x in nodes]
          tensor_xl:torch.tensor = torch.from_numpy(np.array(xl))
          return tensor_xl

    def register_name(self,name,m):
        exec("self." + name + "=m")
        self.layers[name] = m


    def get_name(self, basename:str, m)->str:
        '''this function autogenerates names for unnamed layers like Linear1, Linear2 etc'''
        if basename in self.naming:
            curnumber = max(self.naming[basename])
            self.naming[basename].append(curnumber+1)
            newname = basename+str(curnumber+1)
        else:
            curnumber = 1
            self.naming[basename] = [1]
            newname = basename+str(curnumber)

        exec("self." + newname + "=m")

        self.layers[newname] = m

        #print(newname)
        return newname


    def record(self, funcname:str, param_list:List[str], param_names:List[str])->None:
        '''records function call'''
        if self.mode =='design':
         call:str = "self." + funcname + '('
         p = list(zip(param_list,param_names))
         for x in p:
             if type(x[1]) is str:
                call = call + x[0] + '="' + str(x[1]) +'",'
             elif type(x[1]) is list:
                arg_format = "["+ ",".join(["'" +y +"'" for y in x[1]]) + "]"
                call = call + x[0] + '=' + arg_format +','
             else:
                call = call + x[0] + '=' + str(x[1]) +','
         if call[-1]==',':
           call = call[:-1]
         call = call + ')'
         self.code  = self.code + call + "\n"


    #differentiable operations
    def Flatten(self, target=None, source=None):
        '''flatten operation'''
        if source is None:
            source = self.last_call
        target = source
        if len(self)>0:
         total_size:int = tin_size(self.metatensor[self.last_call])
         self.res  = self.metatensor[self.last_call].view(-1,total_size)
         self.metatensor[target] = self.res
        self.record("Flatten",['target', 'source'],[target, source])
        self.last_call = target
        return self

    def Linear(self, size:int = 50, input_size=None, target=None, source=None):
        ''' linear layer '''
        if source is None:
            source = self.last_call
        #handles proper reshape


        if source in self.layers and len(self)>0:
            prev_layer = self.layers[source]
    #        print(type(prev_layer))
            if type(prev_layer) is torch.nn.LSTM:
               self.metatensor[self.last_call] = self.metatensor[self.last_call].contiguous().view(self.metatensor[self.last_call].size()[1], -1)

        if self.mode=="design":
           if input_size is None:
               input_size = tin_size(self.metatensor[source])
           m = nn.Linear(input_size, size)
           #self.last_call = target
           #input_size
           if target is None:
              target = self.get_name("Linear",m)

           else:
              self.register_name(target, m)


           self.record("Linear",['size','source','target', 'input_size'],[size,source,target,input_size])

        else:

            m = self.layers[target]
        if len(self)>0:
           self.res = m(self.metatensor[source])

           self.metatensor[target] = self.res
        self.last_call = target
        return self

    def dSub(self,target=None, source = []):
        '''computes difference of two source tensors'''
        if target is None:
            target = self.last_call
        if source is None:
            source = self.last_call
        if self.mode=="design":
           self.record("dSub",['source','target'],[source, target])
           if len(source)!=2:
               print("Error: Subtract must have exactly two source arguments!")
               return

        if len(self)>0:

           self.res = (self.metatensor[source[0]] - self.metatensor[source[1]])

           self.metatensor[target] = self.res
        self.last_call=target
        return self

    def dMul(self,target=None, source = []):
        '''computes element wise multiplication of n source tensors'''
        if target is None:
            target = self.last_call
        if source is None:
            source = self.last_call
        if self.mode=="design":
           self.record("dMul",['source','target'],[source, target])
           if len(source)<2:
               print("Error: Multiply  need more then 1 argument!")
               return

        if len(self)>0:
           self.res = self.metatensor[source[0]]
           for x in  source[1:]:
              self.res = (self.res* self.metatensor[x])

           self.metatensor[target] = self.res
        self.last_call=target
        return self

    def Sigmoid(self,target=None, source = None):
        if target is None:
            target = self.last_call
        if source is None:
            source = self.last_call
        if self.mode=="design":
           self.record("Sigmoid",['source','target'],[source, target])

        if len(self)>0:
           f = torch.nn.Sigmoid()
           self.res = f(self.metatensor[source])

           self.metatensor[target] = self.res
        self.last_call=target
        return self


    def Tanh(self,target=None, source = None):
        if target is None:
            target = self.last_call
        if source is None:
            source = self.last_call
        if self.mode=="design":
           self.record("Tanh",['source','target'],[source, target])

        if len(self)>0:
           f = torch.nn.Tanh()
           self.res = f(self.metatensor[source])

           self.metatensor[target] = self.res
        self.last_call=target
        return self

    def Probability(self, sent,start_from=-1):
        self.clear()
        self.append(Node(self.parent_graph,{'chunk':sent}))
        self.Run()
        self.mode = 'predict'
        keys = list(self.metatensor.keys())
        for x in keys:
             if "_state" in x:
                 self.metatensor.pop(x)
#        value_tensor = self.metatensor['Classify1'].detach().numpy()
        if self.device is None:
         value_tensor = self.metatensor['Classify1'].detach().numpy()
        else:
         value_tensor = self.metatensor['Classify1'].detach().cpu().numpy()

        index_tensor = self.res_vector
        p = 1.0
        for i,x in enumerate(value_tensor):
            if i+1 < len(value_tensor) and i>start_from:
                p = p * x[self.metatensor['chunk_tensor'][0][i+1]]
        return p

    def Perplexity(self, sent):
        self.clear()
        keys = list(self.metatensor.keys())
        for x in keys:
             if "_state" in x:
                 self.metatensor.pop(x)
        self.append(Node(self.parent_graph,{'chunk':sent}))
        self.Run()
        self.mode = 'predict'
        if self.device is None:
         value_tensor = self.metatensor['Classify1'].detach().numpy()
        else:
         value_tensor = self.metatensor['Classify1'].detach().cpu().numpy()

        index_tensor = self.res_vector
        p = 1.0
        for i,x in enumerate(value_tensor):
            if i+1 < len(value_tensor):
                p = p + (math.log(x[self.metatensor['chunk_tensor'][0][i+1]],2))
                #p = p * (x[self.metatensor['char_vector'][0][i+1]])
        return math.pow(2,-1/len(sent)*p)




    def Unfold_aux(self, source=None, target=None,start=[0], func='chunk_tensor', seq_length=20):
         '''source - tensor with network state'''

        #for node in self:
         #remove all state vectors
         #print("Unfold")
         #self.disableGPU()
         #print("MOVED TO CPU")
         keys = list(self.metatensor.keys())
         for x in keys:
             if "_state" in x:
                 del self.metatensor[x]
                 #self.metatensor.pop(x)
         result=[]
         torch.cuda.empty_cache()
         #if source is not zero,
         #self.metatensor['_' + source] = self.metatensor[source]
         #print(start)
         for i in range(0,seq_length):
            self.metatensor[func] = torch.tensor(np.array(start), requires_grad=False)
            if self.device is not None:
                self.metatensor[func] = self.metatensor[func].to(self.device)
            self.mode='generate'
            self.Run()
            #print(self.res)
            #del self.metatensor[func]
            #del self.metatensor['LSTM1']
            start = [self.res_vector[len(start)-1]]
            result.append(start[0])
#            self.metatensor['Classify1'].detach()
            torch.cuda.empty_cache()

         #print("END")
         keys = list(self.metatensor.keys())

         for x in keys:
             if "_state" in x:
                 self.metatensor.pop(x)
         #self.enableGPU()
         torch.cuda.empty_cache()
         #print("MOVED TO GPU")
         return result


    def Lookup(self, source = None, target = None, freeze=True,size=30):
        '''Embedding layer'''

        #if target is None:
        #    target = self.last_call
        if source is None:
            source = self.last_call


        if self.mode=="design":
           if self.Text.vecs is not None:
              m = nn.Embedding.from_pretrained(torch.from_numpy(self.Text.vecs.projection_matrix()).float(),freeze=freeze)
           else:

               alphabet = self.resources['alphabet'+source]
               print(len(alphabet)+1)
               m = nn.Embedding(len(alphabet)+1, size)

          # m.load_state_dict({'weight': self.Sentence.vecs.projection_matrix()})
#           m.weights = torch.from_numpy(self.Sentence.vecs.projection_matrix())

           if target is None:
              target = self.get_name("Lookup",m)

           else:
               self.register_name(target,m)
           self.record('Lookup',['source', 'target','size'],[source, target, size])
        else:
           m = self.layers[target]
        if  source in self.metatensor:
         #print("INPUT IS ",source)
         self.metatensor[target] = m(self.metatensor[source])
         self.last_call = target
         self.res = self.metatensor[target]
        return self

    def BLSTM(self, size:int=25, source = None, target = None, input_size=None):
        '''blstm layer'''


        if source is None:
            source = self.last_call

        if source in self.metatensor:
         shape = self.metatensor[source].size()
         if len(shape)==2:
                       #LSTM shape needs 3 dimensions, adding batch dim by defaule
                     #  print("conversion")
                       self.metatensor[source] = self.metatensor[source].unsqueeze(0)
                       #print(shape(self.metatensor[source])
        else:
            print("Warning: no data in model found in BLSTM Layer")


        if self.mode=="design":
           #print(dim_size(self.metatensor[source]))
           #print(self.metatensor[source].size())
           if input_size is None:
               if not source in self.metatensor:
                   print("Error: No data in the model and no input_size specified")
                   return
               input_size = dim_size(self.metatensor[source])
           m = nn.LSTM(input_size, size, bidirectional = True, batch_first=True)
           if target is None:
               target = self.get_name("BLSTM",m)
           else:
               self.register_name(target, m)
           print(target)
           self.record("BLSTM",['size','source','target','input_size'],[size,source,target,input_size])
        else:
             #print(shape)

            m = self.layers[target]
        if source in self.metatensor:
          self.metatensor[target], hidden = m(self.metatensor[source])

          self.res = self.metatensor[target]
        self.last_call = target
        return self

    def SRU(self, size:int=25, source = None, target = None, name=None, state_vector="default", input_size=None,carry_state="no",depth=1):
            '''lstm layer'''
            def repackage_hidden(h):                                                                           
                """Wraps hidden states in new Tensors, to detach them from their history."""                     
                if isinstance(h, torch.Tensor):                                                                  
                    return h.detach()                                                                              
                else:                                                                                            
                    return tuple(repackage_hidden(v) for v in h) 

            def repackage_hidden(h):
              """Wraps hidden states in new Tensors, to detach them from their history."""
              if isinstance(h, torch.Tensor):
                return h.detach()
              else:
                return tuple(repackage_hidden(v) for v in h)

            if source is None:
                source = self.last_call

            if source in self.metatensor:
             shape = self.metatensor[source].size()
             if len(shape)==2:
                           #LSTM shape needs 3 dimensions, adding batch dim by defaule
                         #  print("conversion")
                           self.metatensor[source] = self.metatensor[source].unsqueeze(0)
                           #q =  (self.metatensor[source].size()[0] * self.metatensor[source].size()[1]) * (self.batch_size * self.metatensor[source].size()[1])
                           #print(q)
                           #print(self.metatensor[source].size())
                           #self.metatensor[source] = self.metatensor[source].unsqueeze(0)
                                                    
                           #self.metatensor[source] = self.metatensor[source].view(self.batch_size,self.metatensor[source].size()[2],q)
                          
                           #print(shape(self.metatensor[source])
            else:
                print("Warning: no data in model found in LSTM Layer")


            if self.mode=="design":
               #print(dim_size(self.metatensor[source]))
               #print(self.metatensor[source].size())
               if input_size is None:
                   if not source in self.metatensor:
                       print("Error: No data in the model and no input_size specified")
                       return
                   input_size = dim_size(self.metatensor[source])
               if name is None:
            #    from torchsru import SRU
                m = SRU(input_size, size, bidirectional = False, num_layers=depth)
                name = target
               else:
                m = self.layers[name]

               if target is None:
                   target = self.get_name("SRU",m)
               else:
                   self.register_name(target, m)
               if name is None:
                  name=target
               print(target)
               self.record("SRU",['size','source','target','input_size','name','state_vector','carry_state','depth'],[size,source,target,input_size,name,state_vector,carry_state,depth])
            else:
                 #print(shape)
                
                m = self.layers[name]
            if len(self)==0:return
            self.metatensor[source] = self.metatensor[source].permute(1,0,2)
            if source in self.metatensor:
              if not target+'_state' in self.metatensor:
                if state_vector=="default":
                  self.metatensor[target], hidden = m(self.metatensor[source])
                else:
                  self.metatensor[target], hidden = m(self.metatensor[source], self.metatensor[state_vector])

              else:
                if state_vector=="default":
                  self.metatensor[target], hidden = m(self.metatensor[source],self.metatensor[target+'_state'])
                else:
                  self.metatensor[target], hidden = m(self.metatensor[source],self.metatensor[state_vector])
  
              if carry_state=='yes':
                   self.metatensor[target+'qstate'] = hidden
              if self.mode=='predict' or self.mode=='generate':
                 self.metatensor[target+'_state'] = repackage_hidden(hidden)
        #         print("hidden stored")
#              self.res = self.metatensor[target]
            self.last_call = target
            self.metatensor[target] = self.metatensor[target].permute(1,0,2)
        
            return self


    def LSTM(self, size:int=25, source = None, target = None, name=None, state_vector="default", input_size=None,carry_state="no"):
            '''lstm layer'''

            def repackage_hidden(h):
              """Wraps hidden states in new Tensors, to detach them from their history."""
              if isinstance(h, torch.Tensor):
                return h.detach()
              else:
                return tuple(repackage_hidden(v) for v in h)

            if source is None:
                source = self.last_call

            if source in self.metatensor:
             shape = self.metatensor[source].size()
             if len(shape)==2:
                           #LSTM shape needs 3 dimensions, adding batch dim by defaule
                         #  print("conversion")
                           self.metatensor[source] = self.metatensor[source].unsqueeze(0)
                           #q =  (self.metatensor[source].size()[0] * self.metatensor[source].size()[1]) * (self.batch_size * self.metatensor[source].size()[1])
                           #print(q)
                           #print(self.metatensor[source].size())
                           #self.metatensor[source] = self.metatensor[source].unsqueeze(0)
                                                    
                           #self.metatensor[source] = self.metatensor[source].view(self.batch_size,self.metatensor[source].size()[2],q)
                          
                           #print(shape(self.metatensor[source])
            else:
                print("Warning: no data in model found in LSTM Layer")


            if self.mode=="design":
               #print(dim_size(self.metatensor[source]))
               #print(self.metatensor[source].size())
               if input_size is None:
                   if not source in self.metatensor:
                       print("Error: No data in the model and no input_size specified")
                       return
                   input_size = dim_size(self.metatensor[source])
               if name is None:
                m = nn.LSTM(input_size, size, bidirectional = False, batch_first=True)
                name = target
               else:
                m = self.layers[name]

               if target is None:
                   target = self.get_name("LSTM",m)
               else:
                   self.register_name(target, m)
               if name is None:
                  name=target
               print(target)
               self.record("LSTM",['size','source','target','input_size','name','state_vector','carry_state'],[size,source,target,input_size,name,state_vector,carry_state])
            else:
                 #print(shape)
                
                m = self.layers[name]
            if source in self.metatensor:
              if not target+'_state' in self.metatensor:
                if state_vector=="default":
                  self.metatensor[target], hidden = m(self.metatensor[source])
                else:
                  self.metatensor[target], hidden = m(self.metatensor[source], self.metatensor[state_vector])

              else:
                if state_vector=="default":
                  self.metatensor[target], hidden = m(self.metatensor[source],self.metatensor[target+'_state'])
                else:
                  self.metatensor[target], hidden = m(self.metatensor[source],self.metatensor[state_vector])
  
              if carry_state=='yes':
                   self.metatensor[target+'qstate'] = hidden
              if self.mode=='generate':
                  self.metatensor[target+'_state'] = repackage_hidden(hidden)
        #         print("hidden stored")
#              self.res = self.metatensor[target]
            self.last_call = target
            return self


    def Classify(self,target=None, class_target=None, cmode="Normal", source=None, input_size = None):
        ''' make classification layer'''
        #TODO: this needs simplification
        global res
        if source is None:
            source = self.last_call
        if class_target is None:
            class_target = self.class_target
        if class_target is None:
            print("ERROR: No classes loaded")
            return

        #shape assignment
        if len(self)>0:
         if source in self.layers:
             prev_layer = self.layers[source]
             if (type(prev_layer) is torch.nn.LSTM) or  (type(prev_layer) is SRU):
                    self.metatensor[source] = self.metatensor[source].contiguous().view(unisize(self.metatensor[source]), -1)
                #    print("size conversion", self.metatensor[source].size())
        criterion = nn.CrossEntropyLoss()

        if self.mode=='design':
            self.class_target = class_target
            if len(self)>0 and len(self.classes)==0:
               self.classes = self.Distinct(class_target)

            if input_size is None:
               input_size = tin_size(self.metatensor[source])
            print(len(self.classes))
            print(input_size)

            m = nn.Linear(input_size, len(self.classes))
            if target is None:
              target = self.get_name("Classify",m)
            else:
               self.register_name(target, m)
            self.target = target

            maxsize = min(len(self),self.batch_size)
            #if cmode == "Normal":
            print("c",class_target)
            if class_target!="_seq_model":
               print(class_target,"target")
               class_vector = self.vectorize_classes(self[0:maxsize], class_target)
            else:
               if len(self)>0:
                 class_vector = self.metatensor['classes'].contiguous().view(unisize(self.metatensor[source]))
            if len(self)>0:
             s = nn.Softmax(dim=-1)
             self.res = s(m(self.metatensor[source]))
            #print("Classification loss:" + str(self.loss))
            self.record("Classify",['target','class_target','cmode','source', 'input_size'],[target,class_target,cmode, source, input_size])
            print("modes", self.target, self.class_target, len(self))



        if self.mode =='train' or self.mode == 'eval':
            maxsize = min(len(self),self.batch_size)

            m = self.layers[target]
            if class_target!="_seq_model":
               class_vector = self.vectorize_classes(self[self.index:self.index+self.batch_size], class_target)
            else:
               class_vector = self.metatensor['classes'].contiguous().view(unisize(self.metatensor[source]))
               #print("wrong_class")
            if self.device is not None:
               class_vector = class_vector.to(self.device)
            #print(class_vector)
         #   print("maxsize",maxsize, self.index, len(class_vector),len(self))
         #   print("cwords", [x["слово"] for x in self[self.index:self.index+self.batch_size]])
                #print("classification",self.index,self.batch_size,len(self))

            self.res = m(self.metatensor[source])
            self.metatensor[target]=self.res
        #    print(self.res.size())
    ##        print(class_vector.size())
#            print(self.res)
#            print(torch.min(class_vector))

            self.loss = criterion(self.res, class_vector)

            s = nn.Softmax(dim=-1)
            self.res = s(self.res)
            if self.device is None:
               n = self.res.detach().numpy()
            else:
               n = self.res.detach().cpu().numpy()

            #print(n)
            n = list(np.argmax(n, axis = 1))

            #print(self.classes)
            self.res = [self.classes[x] for x in n]
            self.result = self.res


        if self.mode == 'predict' or self.mode=='generate':
            m = self.layers[target]
            self.res = m(self.metatensor[source])

            s = nn.Softmax(dim=-1)

            self.res = s(self.res)
            self.metatensor[target]=self.res
            if self.device is None:
              n = self.res.detach().numpy()
            else:
              n = self.res.cpu().detach().numpy()
            #print(n)
            n = list(np.argmax(n, axis = 1))

            #print(self.classes)
            self.res_vector = n
            self.res = [self.classes[x] for x in n]
            #write classification result to node
            i = self.index#-1
           # print(self.res)
            #print(len(self.res))
            #print(len(self))
            #print(i)
            try:
             for x in self.res:
                self[i][target] = x
    #           print(x)
                i = i + 1
            except:
                pass
              #  print("error!")
            self.result = self.res
            return self.res

        self.last_call=target
        return self

    #accuracy evaluation routines

    def Accuracy(self, target, prediction):
        '''accuracy of classification'''
        count = 0.0
        total = 0.0
        for node in self:
            if prediction  in node:
                    if node[prediction] == node[target]:
                        count = count + 1
            total = total + 1
        return count / total

    #not differentiable aux operations
    def Rename(self, source:str, target:str):
        '''renames field source into field target'''
        if self.mode =="design":
            self.record('Rename',['source','target'],[source, target])
        #print(self.mode)
        if self.mode=='predict':
           # print(0/0)
            for i  in range(len(self)):
                if source in self[i]:
                 self[i][target] = self[i][source]
                 self[i].dict.pop(source)
                #print(self[i][target],target)

    def Match(self, node):
        if self.mode =="design":
            self.record('Match',['node'],[node])
            return self
        if self.mode=='predict':
            return super(dNodeList,self).Match(node)

    def Shuffle(self):
        random.shuffle(self)
        return self

    def ParentModel(self):
        self.rstate = 'Parent'
        return self.Parent

    def Exec(self):
        self.Run()
        if self.Child[0].mode=='design':
         self.Child[0].record("Parent[0].Exec",[],[])
        return self
    
    def DistributeFeatures(self, source="", target=""):
        if len(self)>0:
         node = self[self.index]
         children = node.Children({})
         p = []
         for i,x in enumerate(node[source+'_inds_']):
             p.append(self.metatensor[target][x])
             
         #res = [x[target] for x in children]
         self.Child[0].metatensor[target] = torch.stack(p,dim=0).detach()
         
        if self.Child[0].mode=='design':
           self.Child[0].record('Parent[0].DistributeFeatures',['source','target'],[source, target]) 
       # else:
       #     self.next()
        return self

    def LoadState(self,filename):                                               
        self.load_state_dict(torch.load(filename))   

    def Unroll(self):
        graph = self.parent_graph
        lst = dNodeList(graph)
        lst.save_func = self.save_func
        for x in self:
            words = x.Children({})
            for word in words:
                lst.append(word)
        lst.Parent = [self]
        self.Child = [lst]
        self.batch_size = 1 #TODO: hack, to be removed
        return lst

    def ConvertTo(self, ctype="sentence"):
        graph = self.parent_graph
        lst =  dNodeList(graph)
        units = graph.Match({'type':ctype})
        for x in units:
            lst.append(x)
        return lst

    def LoadContent(self, source):
        '''loads content from different nodelist into this nodelist'''
        self.parent_graph = source.parent_graph
        for x in source:
            
            self.append(x)


    #vectorization primitives
    def Features(self, funcname=None, source=None, target=None):
        '''generates features using func'''
        if source is None:
            source = self.last_call
        #func = eval(funcname)
        if self.mode=='design':
            self.record('Features',['source','target'],[source,target])
        if len(self)>0:
         if self.mode=='design':
             #build feature list
             d = {}
             all_features_list = list(set(reduce(lambda x,y:x+y,[funcname(self,index,source) for index in range(len(self))])))
             #for speed of access will make a dictionary
             for i,x in enumerate(all_features_list):
                 d[x] = i

             self.features[target] = d
             if funcname is not None:
               self.func = funcname

         #vectorize batch
         nodes = self[self.index:self.index+self.batch_size]
         #if self.mode=='predict':
        #        nodes = self
         for i in range(len(nodes)):
             cur_features = self.func(nodes, i,source)
             vector = np.zeros(len(self.features[target]))
             allf=self.features[target]
             for x in cur_features:
                 if x in cur_features:
                     ind = allf[x]
                     vector[ind] = 1.0


             nodes[i][target] = torch.from_numpy(vector).float()
        # print("NODES=",len(nodes))
         res = [x[target] for x in nodes]
         self.metatensor[target] = torch.stack(res,dim=0)
        self.last_call = target
        return self

    #def AddDim(source=None,target=None)

    def dMerge(self, source=[], target=None):
        if self.mode=='design':
             if len(source)==0:
                 print("Error: No source specified for dMerge")
                 return
             self.record('dMerge',['source','target'],[source,target])
             for x in source:
                 if not x in self.metatensor:
                     print("Design mode warning: " + x + "is not yet computed")
        output_tensor = None
        if source[0] in self.metatensor:
         #execute merge if we have necessary data computed
       #  print(len(self.metatensor[source[0]].shape),len(self.metatensor[source[1]].shape),len(source))
         if len(self.metatensor[source[0]].shape)==2 and len(self.metatensor[source[1]].shape)==3 and len(source)==2:
             if self.mode=='design':
              print("WARNING: 2d to 3d merge defaults to sequence merge now, 2d data will be padded")
             output_tensor = torch.cat([self.metatensor[source[0]].unsqueeze(0),self.metatensor[source[1]]],2)

         if len(self.metatensor[source[0]].shape)==2 and output_tensor is None:
                 #merge 2d data (batch*data)
                 input_tensors = [self.metatensor[x] for x in source]
                 output_tensor = torch.cat(input_tensors,1)
         if len(self.metatensor[source[0]].shape)==3 and output_tensor is None:
                 #merge 3d data (batch*word*word_content)
                 input_tensors = [self.metatensor[x] for x in source]
                 output_tensor = torch.cat(input_tensors,2)
         if output_tensor is None:
             print("ERROR: Operands shape is currently not supported")
         self.metatensor[target] = output_tensor
         return self

    #differential programming primitives
    def dContextMap(self, source:str, target:str, size:int=25):
                   dtype = None
                   if source in self[0]:
                      dtype = type(self[0][source])
                   if source in self.metatensor:
                       dtype = type(self.metatensor[source])

                   if dtype is str:
                       text = self[0][source]
                       if 'а' in text or 'о' in text or 'и' in text or 'е' in text:
                            lang = 'ru'
                            print("selected language: ru")
                       else:
                            print("selected language: en")
                            lang = 'en'

                       if lang=='ru':
                            self.Text.VectorizeLookup(source=source, embeddings='ru_50c')
                       else:
                            self.Text.VectorizeLookup(source=source, embeddings='en_100')
                       self.Lookup()
                       source = self.last_call
                      # print(self.last_call)
                   cur_tensor = self.metatensor[source]
                   shape = cur_tensor.size()

                   #self.Linear(size=20).Sigmoid().BLSTM(size=size)
                   self.BLSTM(size=size,source=source).Sigmoid()
                   self.Classify(class_target=target,target='dMap1')
                   self.Rename(source='dMap1',target=target)
                   return self

    def dMap(self, source:str, target:str, size:int=256):
        '''dMap is generic operation that corresponds to any differntiable module that transforms fixed size vector into another fixed size vector
           it is also automaticall vectorizes data types which it knows how to vectorize (image, word, sequence of words) but you can apply dMap after
           your own feature generation function (currently unimplemented)
           Args:
            source: field from  data is taken
            target: field were to write result (also source of classes for training in design/complie time)
        '''
        #determine data type for vectorization (assumes each field has consistent datatype)
        #TODO: verify consistency of data
        if self.mode=="design":
           dtype = None
           if source in self[0]:
              dtype = type(self[0][source])
           if source in self.metatensor:
               dtype = type(self.metatensor[source])
           print(dtype)
           if dtype is None:
               print("dMap Error: invalid source specified")
               return self


           if dtype is str:
               #determine language to be used
               text = self[0][source]
               if 'а' in text or 'о' in text or 'и' in text or 'е' in text:
                   lang = 'ru'
               else:
                   lang = 'en'

               if len(self[0][source].split())==1:
                   #for single word case
                   if lang =='ru':
                      self.Text.VectorizeWordSimple(source=source, embeddings='ru_50c')
                   else:
                      self.Text.VectorizeWordSimple(source=source, embeddings='en_100')
               else:
                   #for multi word text
                   print("multipart")
                   if lang =='ru':
                      self.Text.VectorizeNbowSimple(source=source, embeddings='ru_50c')
                   else:
                      print("en")
                      self.Text.VectorizeNbowSimple(source=source, embeddings='en_100')

                   if not target in self[0]:
                    self.Linear(size=size, target=target).Sigmoid()
                   else:
                    self.Linear(size=size).Sigmoid()
                    self.Classify(class_target=target,target='dMap1')
                    self.Rename(source='dMap1',target=target)
                   return self

           if dtype is mvImage:
                self.Image.Vectorize(source=source).model.Flatten()

                if not target in self[0]:
                   self.Linear(size=size, target=target).Sigmoid()
                else:
                   self.Classify(class_target=target,target='dMap1')
                   self.Rename(source='dMap1',target=target)
                return self

           #for tensor types (continuation of dMap chain)
           if dtype is torch.Tensor:
              if target in self[0]:
                  #Final dMap to class
                  self.Classify(class_target=target,target='dMap1',source=source)
                  self.Rename(source='dMap1',target=target)
              else:
                  #intermidiate dMap to tensor
                  print(source)
                  self.Linear(size=size, target=target, source=source).Sigmoid()
              return self
        print("dMap Error: Data type is not supported")
        return self







    #compilation routines
    def Run(self):
        '''executes all recordered model code'''
        exec(self.code)

    def next(self):
        '''this function moves model to the next batch'''
        if self.index + self.batch_size < len(self):
            self.index = self.index + self.batch_size
        else:
            self.index = 0

    def Clone(self, split=1.0, empty=False):
        '''clone makes a copy of model and data, optionally splitting data into test/train set, according to split proportion'''
        model2 = dNodeList(self.parent_graph)

        model2.code = self.code
        model2.Child = self.Child
        if self.Parent is not None:
            model2.Parent = self.Parent
        model2.class_source =  self.class_source
        model2.layers = self.layers
        model2.device = self.device
        model2.naming = self.naming
        model2.resources = self.resources
        model2.target = self.target
        model2.features = self.features
        model2.func = self.func
        model2.save_func = self.save_func
        model2.class_target = self.class_target
        #model2.start_point = self.start_point
        #model2.mtype = self.mtype
        model2.metatensor = self.metatensor
        model2.Text.vecs = self.Text.vecs
        #model2.classes = self.classes
        model2.index = 0
        model2.classes = self.classes
        model2.batch_size = self.batch_size
        #copy data content with split
        if not empty:
         copy_nodes = self[int(len(self)*split):]
         if self.Parent is not None:
          en_copy=False
         else:
             en_copy=True
         for x in copy_nodes:
             if self.Parent is not None:
                 real_c = x.Parents({})[0].Children({})
                 if real_c[0] == x:
                     en_copy=True
             if en_copy:
              model2.append(x)
         #remove original data from self
         print(len(copy_nodes))
         del self[int(len(self)*split):]
         #for x in copy_nodes:
         #    self.remove(x)


        return model2

    def Predict2(self, prediction_target=None, target=None, basemode='train'):
            '''this is aux predict when train function. need fixing'''
            if prediction_target is None:
               prediction_target = self.target
            #print("writing",prediction_target, target)

            #self.mode = 'train'
           # self.Run()
  #          self.f1 =""
            i = 0
           # print('-----')
            self.index = 0
            self.total_loss = 0.0
            while self.index<len(self):
                self.mode = basemode
                self.Run()
                if basemode=='train':
                   self.total_loss = self.total_loss + self.loss.detach().item()
                   try:
                    for x in self.result:
                     self[i][prediction_target] = x
                     i = i + 1
                   except:
                       target =None
                     #  print("fail")
                       pass
    #               print(x)
                     
                self.index = self.index + self.batch_size
                print(str(self.index)+"          ",end='\r',flush=True)
            self.index = 0
           # print(target)
            if target is not None and (not 'classes' in self.metatensor):
               self.accuracy = self.Accuracy(target, prediction_target)
               #print(self.accuracy)
            else:
               self.accuracy = None
            return self.result

    def predict(self, data):
        new_list = self.Clone(empty=True)
        for x in data:
            new_list.append(x)
        new_list.Predict2(basemode='predict')
        return data


    def compile(self,  opt="Adam", size=50, lr=0.01,avg_steps = None,thresold = 0.002, post_func = None, weight_decay=3e-4):
        _compile(self, opt, size, lr, avg_steps=avg_steps, thresold=thresold, post_func=post_func, weight_decay=weight_decay)
        return self.predict

    def LoadState(self,filename):
        self.load_state_dict(torch.load(filename))

    def Export(self, filename):
            if filename=="model":
                print("ERROR: model is reserved keyword, please choose another name")
                return
            os.mkdir(filename)
            self.load_state_dict(torch.load("tempmodel.tmp"))
            torch.save(self.state_dict(), "./" + filename + "/" + filename + ".mod")
            code = self.code
            cbase = []
            cbase.append("from neuthink import metagraph as m")
            cbase.append("from neuthink.textviews import SentenceView")
            cbase.append("from neuthink.graph.basics import Graph")
            cbase.append("import torch")
            cbase.append("import os")
            cbase.append("graph = Graph()")
            #cbase.append("import torch")
            cbase.append("path = os.path.dirname(os.path.abspath(__file__))")
            cbase.append("print(path)")

            cbase.append("model = m.dNodeList(graph)")
            cbase.append("model.classes = [x.strip() for x in open(path+'/'+'" + filename + ".cls','r').readlines()]")
            if len(self.features)>0:
                #import feautures
                cbase.append("import json")
                cbase.append("model.features = json.loads(open(path+'/'+'" + filename + ".fea','r').read())")
                cbase.append("from ." + self.func.__module__ + " import " + self.func.__name__)
                cbase.append("model.func = " + self.func.__name__)
            if len(self.resources)>0:
                cbase.append("import json")
                cbase.append("model.resources = json.loads(open(path+'/'+'" + filename + ".res','r').read())")


            for x in code.split('\n'):
                cbase.append(x.replace('self','model'))
            cbase.append("model.load_state_dict(torch.load(path +'/'+'"+ filename + ".mod'))")
            cbase.append("model.mode='predict'")
            cbase.append(filename +" = lambda x: model.predict(x)")
            f = open("./" + filename +'/__init__.py','w')
            f.writelines([x+"\n" for x in cbase])
            #save classes
            f = open("./" + filename +"/" + filename + '.cls','w')
            for x in self.classes:
                f.write(x+'\n')
            #export features
            if len(self.features)>0:
                feat_string = json.dumps(self.features)
                f = open("./" + filename +"/" + filename + '.fea','w')
                f.write(feat_string)
                f.close()

                feat_module_name = sys.modules[self.func.__module__].__file__
                copyfile(feat_module_name, "./" + filename +"/")
            #export resources
            if len(self.resources)>0:
                feat_string = json.dumps(self.resources)
                f = open("./" + filename +"/" + filename + '.res','w')
                f.write(feat_string)
                f.close()


def test_decode(model):
    data = model.Unfold_aux(start=model.Text.StringEncode('З'), seq_length=285)
    data = model.Text.StringDecode(data)
    model.mode = 'train'
    return data

def _compile(model:dNodeList, opt="Adam", size=50, lr=0.01,avg_steps = None,thresold = 0.002, post_func=None, weight_decay=3e-4):


    error = 900000000000000
#    thresold = 0.002
    prev_error = 10000000
    test_f1 = ""
    best_f1 = ""
    accuracy = None
    best_accuracy = None
    moving_avg = 0
    best_loss = 9000
    merror = 100000
#    avg_steps = 550
    counter = 0
    model.mode = "train"
    print(model.save_func)
    if avg_steps is None:
        avg_steps = len(model)/size
    model.batch_size = size
    if opt=="SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = lr ,momentum=0.8, nesterov=True)
        #optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 3e-4)
    if opt=="Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
        #optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    model.index = 0
    q = len(model)
    if len(model)<450000:
       test_model = model.Clone(split=0.85)
    else:
        #test_model = model.Clone(split=0.9972)
        test_model = model.Clone(split=0.985)
    model.test_set  = test_model
    stats = open('stats.csv','w')
    stats.write('iteration \t train_error \t test_error \n')

    step  = 0
    while lr > thresold:

        model.Run()
        optimizer.zero_grad()
        batch_error = model.loss
        batch_error.backward()
        optimizer.step()
        model.next()
        #print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh888")

        display_comp_status(counter, batch_error.detach().item(), error, test_f1, best_loss, best_f1, lr,test_accuracy=accuracy, best_accuracy = best_accuracy,train_moving_loss=merror)
        counter = counter + 1
        moving_avg = moving_avg + batch_error.detach().item()
        if counter > avg_steps:
           prev_error = error
           merror = moving_avg #/ avg_steps
           moving_avg = 0

           test_model.Predict2(target=test_model.class_target)
           #print("test error", test_model.total_loss)
           #print(test_model.f1)
           if hasattr(test_model, "test_f1"):
            test_f1 = test_model.f1
           if hasattr(test_model, "accuracy"):
               accuracy = test_model.accuracy
           if model.save_func is not None:
               print("save called!")
               model.save_func(model, 'train.txt')
               test_model.save_func(test_model, 'test.txt')
           step = step + 1
           error = test_model.total_loss
           stats.write(str(step) + '\t' + str(merror) + '\t' + str(error) + '\n' )

           if best_loss > error:
               best_loss = error
               best_f1 = test_f1
               best_accuracy = accuracy
               torch.save(model.state_dict(), "tempmodel.tmp")


           if prev_error<error:
            lr = lr * 0.95
            #optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-5)

        #    print("new lr is ",lr)
            #optimizer = torch.optim.SGD(model.parameters(), lr = lr ,momentum=0.9)

            for param_group in optimizer.param_groups:
               param_group['lr'] = lr
          
           counter = 0
           if post_func is not None:
                r =  post_func(model)
                if r!=None:
                    f = open("post_results.txt","a")
                    f.write(str(step) + '\t' + r+'\n')
                    f.close()
           torch.cuda.empty_cache()
    print()
    print()
    print()
    stats.close()
    print("*************Results***************")
    print("Test error:" + str(best_loss) + "                     ")
    if best_f1!="":
       print("Test F1:" + str(best_f1) + "                     ")
    if best_accuracy is not None:
       print("Test accuracy:" + str(best_accuracy) + "                     ")
    return model
