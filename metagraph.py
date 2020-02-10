from typing import List
import types
import os
import random
import json
import sys
import torch
import torch.nn as nn
import numpy as np
from functools import reduce
from shutil import copy as copyfile
import math
import copy

try:
    from apex import amp
    from apex.optimizers import FusedAdam
    from apex.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    APEX_ENABLED = True
except ModuleNotFoundError:
    APEX_ENABLED = False
try:
  from sru import SRU
  sru_enabled=True
except:
  print("SRU not installed, SRU layers support disabled")
  sru_enabled=False


from neuthink.graph.basics import Node,NodeList,Graph
from neuthink.nImage import  mvImage
import neuthink.metaimage as mimage
import neuthink.metacluster as mcluster
import neuthink.metatext
from neuthink.display import display_comp_status


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

    def Zip(self, newlist):
        if len(newlist)!=len(self):
            print("Error, list size must match!")
            return
        for i in range(0,len(self)):
            keys = newlist[i].keys()
            for key in keys:
                newkey = (key + "1") if key in self[i] else key
                self[i][newkey] = newlist[i][key]
        return self

    def Set(self,field,value):
       for x in self:
           x.Set(field,value)

       return self

    def Detach(self,node):
       for x in self:
           x.Detach(node)

       return self

    def Join(self):
        newbase = Node(self.parent_graph,{'type':self[0]['type']})
        newlist = NodeList(self.parent_graph)
        for x in self:
            c = x.Children({})
            for x in c:
                newbase.Connect(x)
            x.Delete()
        newlist.append(newbase)
        return newlist

    def Deduplicate(self, key):
        '''slow deduplication code. leaves only elements with unique key field'''
        all_key_values = self.Distinct(key)
       # print(all_keys)
        newlist = NodeList(self.parent_graph)
        self.mode='predict'
        print(len(all_key_values))
        for x in all_key_values:
            node = self.MatchOne({key:x})
            print(key,x)
            newlist.append(node)

        return newlist


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
        self.Cluster = mcluster.MetaCluster(self)
        self.mode = "design"
        self.loadfunc=None
        self.code:str = ""
        self.metatensor={}
        self.res = None
        self.second_result=0
        self.batch_size = 50
        self.index = 0
        self.title='main'
        self.naming = {}
        self.save_func = None
        self.class_source = None
        self.layers = {}
        self.loss = 0
        self.classes=[]
        self.rnns=[]
        self.test_set= None
        self.features={}
        self.func = None
        self.resources={}
        self.Parent = None
        self.Child = None
        self.device=None
        self.target=None
        self.class_target=None
        self.subindex =0

    def  __hash__(self):
        '''for pytorch 1.0 compatibility. not very good idea, but not likely to cause any trouble'''
        return super(torch.nn.Module,self).__hash__()

    def enableGPU(self,device='cuda:1'):
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.device = device
        print(device)
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
        #if not 'LSTM' in name:
        exec("self." + name + "=m")
        #else:
        #  exec("self.rnns.append(m)")

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

    #model allocation primitives
    def ToDevice(self, source=None,device='cpu'):
        if self.mode=='design':

            self.record("ToDevice",['source', 'device'],[source,device])
        else:
            self.metatensor[source] =  self.metatensor[source].to(device)
        return self

        self.metatensor['source'] =  self.metatensor['source'].to(device)

    def PutLayerToDevice(self, layer_name,device):
        self.layers[layer_name].to(device)


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
        if self.mode=='design':
            self.record("Flatten",['target', 'source'],[target, source])
        self.last_call = target
        return self

    def design_register_layer(self, name, target, layer_type,m):
            
           if target is None:
               target = self.get_name(layer_type ,m)

           else:
               self.register_name(target, m)
           
           if name is not None and name not in self.layers:
              if target is None:
               target = self.get_name(name,m)

              else:
               self.register_name(name, m)
            
    def get_stored_layer(self,name, target):
            if name is None:
              m = self.layers[target]
            else:
              m = self.layers[name]        
            return m

    def Pool2D(self,source=None,target=None, kernel_size=(2,2), stride=None, padding=0, name=None):
        '''2D max pooling layer'''
        if source is None:
            source = self.last_call
        if target==None:
           target = source
        m = None
        if self.mode=="design":
          if name is None and not target in self.layers:
            m = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
            self.design_register_layer(name,target,'Pool2D',m)
            self.record("Pool2D",['kernel_size','source','target', 'padding','stride'],[kernel_size,source,target,padding,stride])
        if m is None:
          m = self.get_stored_layer(name,target) 
        if len(self)>0:
           input_data = self.metatensor[source]
           self.res = m(input_data)
           self.metatensor[target] = self.res
        self.last_call = target

    def Upsample(self, source=None,target=None, size=None, scale_factor=None, mode='nearest', align_corners=None, name=None):
        if source is None:
            source = self.last_call
        if target==None:
           target = source
        m = None
        if self.mode=="design":
          if name is None and not target in self.layers:
            m = nn.Upsample(size=size,scale_factor=scale_factor,mode=mode,align_corners=align_corners)
            self.design_register_layer(name,target,'Upsample',m)
            self.record("Upsample",['size','source','target', 'scale_factor','mode','align_corners'],[size,source,target,scale_factor,mode, align_corners])
        if m is None:
          m = self.get_stored_layer(name,target) 
        if len(self)>0:
           input_data = self.metatensor[source]
           self.res = m(input_data)
           self.metatensor[target] = self.res
        self.last_call = target


    def Conv2D(self, kernel_size:int=5,nmaps:int=1,padding:int=0,stride:int=1, input_size=None, target=None, source=None,in_channels=None,name=None):
        ''' 2D convolution layer '''
        if source is None:
            source = self.last_call
        if target==None:
           target = source
        #handles proper reshape

        if source in self.layers and len(self)>0:
            prev_layer = self.layers[source]
    #        print(type(prev_layer))
            if type(prev_layer) is torch.nn.LSTM:
               self.metatensor[self.last_call] = self.metatensor[self.last_call].contiguous().view(self.metatensor[self.last_call].size()[1], -1)
        m = None

        if self.mode=="design":
           #if input_size is None:
           #    input_size = tin_size(self.metatensor[source])
           if in_channels is None:
            in_channels= 1 if len(self.metatensor[source].shape)==3 else self.metatensor[source].shape[1]

           if name is None and not target in self.layers:
            m = nn.Conv2d(in_channels, nmaps, kernel_size, stride=stride, padding=padding)  
            self.design_register_layer(name,target,'Conv2D', m)
         
           self.record("Conv2D",['kernel_size','source','target', 'nmaps','padding','stride','in_channels'],[kernel_size,source,target,nmaps,padding,stride,in_channels])

        if m is None:
            m = self.get_stored_layer(name,target) 

        if len(self)>0:
           input_data = self.metatensor[source]
           if len(input_data.shape)==3:
               input_data = input_data.unsqueeze(1)


           self.res = m(input_data)

           self.metatensor[target] = self.res
        self.last_call = target
        return self



    def Linear(self, size:int = 50, input_size=None, target=None, source=None, data_parallel=False):
        ''' linear layer '''
        if source is None:
            source = self.last_call
        #handles proper reshape

        m = None
        if source in self.layers and len(self)>0:
            prev_layer = self.layers[source]
        #    print(type(prev_layer))
            if type(prev_layer) is torch.nn.LSTM:
        #       self.metatensor[self.last_call] = self.metatensor[self.last_call].contiguous().view(self.metatensor[self.last_call].size()[1], -1)
                self.psize = self.metatensor[self.last_call].shape
        #     
                self.metatensor[self.last_call] = self.metatensor[self.last_call].contiguous().view(-1, self.metatensor[self.last_call].size()[2])
        
            if sru_enabled:
             if type(prev_layer) is SRU:
               self.psize = self.metatensor[self.last_call].shape
        #       print("LIN,", self.psize)
               self.metatensor[self.last_call] = self.metatensor[self.last_call].contiguous().view(-1, self.metatensor[self.last_call].size()[2])
               
        #     print("lin", self.metatensor[self.last_call].shape)
        if self.mode=="design":
           if input_size is None:
               input_size = tin_size(self.metatensor[source])
        #    if len(self.metatensor[source].shape)>2:
        #       self.metatensor[source] = self.metatensor[source].contiguous().view(self.metatensor[source].size()[0], -1)


           m = nn.Linear(input_size, size)
           #self.last_call = target
           #input_size
           if target is None:
              target = self.get_name("Linear",m)


           else:
              if not target in self.layers:
               self.register_name(target, m)
           


           self.record("Linear",['size','source','target', 'input_size'],[size,source,target,input_size])

        if m is  None:

            m = self.layers[target]

        if data_parallel and self.mode=='train' :
                    m = nn.DataParallel(m)
        
        if len(self)>0:
        #    if len(self.metatensor[source].shape)>2:
        #       self.metatensor[source] = self.metatensor[source].contiguous().view(self.metatensor[source].size()[0],self.metatensor[source].size()[1]*self.metatensor[source].size()[2])

    #       print(self.metatensor[source].shape)

           self.res = m(self.metatensor[source])

           self.metatensor[target] = self.res
        self.last_call = target
        return self

    def dAdd(self,target=None, source = []):
        '''computes the sum of two source tensors'''
        if target is None:
            target = self.last_call
        if source is None:
            source = self.last_call
        if self.mode=="design":
           self.record("dAdd",['source','target'],[source, target])
           if len(source)!=2:
               print("Error: Addition must have exactly two source arguments!")
               return

        if len(self)>0:

           self.res = (self.metatensor[source[0]] + self.metatensor[source[1]])

           self.metatensor[target] = self.res
        self.last_call=target
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

    def RelU(self,target=None, source = None):
        if target is None:
            target = self.last_call
        if source is None:
            source = self.last_call
        if self.mode=="design":
           self.record("RelU",['source','target'],[source, target])

        if len(self)>0:
           f = torch.nn.ReLU()
           self.res = f(self.metatensor[source])

           self.metatensor[target] = self.res
        self.last_call=target
        return self

    def PRelU(self,channels = 1, target=None, source = None):
        if target is None:
            target = self.last_call
        if source is None:
            source = self.last_call
        if self.mode=="design":
           self.record("PRelU",['source','target','channels'],[source, target, channels])
           f = torch.nn.PReLU(num_parameters=channels)
           print(channels)
           if target is None:
               target = self.get_name("PRelU",f)
           else:
               self.register_name(target,f)
        else:
           f = self.layers[target]
        

        if len(self)>0:
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

    def _tokenized(self, sent):
        if 'wordpieces' in self.resources:
            print(self.resources['filename'])
            modelname = self.resources['filename'][:self.resources['filename'].rfind('.')]
            print(modelname)
            modelname = modelname + '.model'
            sent = neuthink.metatext.BPETokenize(sent, modelname)
            sent = sent[0].split(' ')
        return sent


    def Probability(self, sent ,start_from:int =-1, init_state={}, keep_state=False)->float:
        '''calculates LM probability for a given sent, starting from position start_from, and initial state init_state'''

        def clear_state_vector(self):
            keys = list(self.metatensor.keys())
            #this removes state vector
            for x in keys:
                if "_state" in x:
                    self.metatensor.pop(x)

        self.clear()
        self.index=0
        if 'wordpieces' in self.resources:
#            print(self.resources['filename'])
            modelname = self.resources['filename'][:self.resources['filename'].rfind('.')]
#            print(modelname)
            modelname = modelname + '.model'
            sent = neuthink.metatext.BPETokenize(sent, modelname)
            sent = sent[0].split(' ')
#            print(sent)

        self.append(Node(self.parent_graph,{'chunk':sent}))
        self.index=0
        self.metatensor=copy.copy(init_state)
        if keep_state:
            self.mode="predict"
        self.Run()
        self.mode = 'predict'
        if not keep_state:
            clear_state_vector(self)

        value_tensor = self.metatensor['Classify1'].detach().cpu().numpy()
        index_tensor = self.res_vector
        p = 1.0
        for i,x in enumerate(value_tensor):
            if i+1 < len(value_tensor) and i>start_from:
                p = p * x[self.metatensor['chunk_tensor'][0][i+1]]
        return p

    def ResetLSTMState(self):
        keys = list(self.metatensor.keys())
        for x in keys:
             if "_state" in x:
                 self.metatensor.pop(x)

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




    def Unfold_aux(self, source=None, cluster_id=0, target=None,start=[0], func='chunk_tensor', seq_length=20, transformer_len = None,sampling="maxprob"):
         '''source - tensor with network state'''
 
         def zeros_append(len_big, len_small):
             if len_big > len_small:
                return list(np.zeros(len_big - len_small))
             else:
                 return []
        
         def exact_sample(s):
             index_list =[]
             for x in s:
                 index, prob = x
                 prob = int(prob*100)
                 index_list = index_list + [index] * prob
#             print(index_list)
             return random.choice(index_list)

         def nucleus_sampling():
             max_p = 0.1
             indexes = self.metatensor['class_sorted'][-1]
             probs = self.metatensor['class_probs'][-1]
#             print(indexes)
             acc_p = 0
             index_list = []
             qind = len(indexes)-1
            
             while acc_p < max_p:
                 index_list.append((indexes[qind], probs[indexes[qind]]))
                 qind = qind - 1
                 acc_p = acc_p + probs[indexes[qind]]
#             print(index_list)
             return exact_sample(index_list)

             

        #for node in self:
         #remove all state vectors
         #print("Unfold")
         #self.disableGPU()
         #print("MOVED TO CPU")
         keys = list(self.metatensor.keys())
#         print(start)
         for x in keys:
             if "_state" in x:
                 del self.metatensor[x]
                 #self.metatensor.pop(x)
         result=[]
         torch.cuda.empty_cache()
         #if source is not zero,
         #self.metatensor['_' + source] = self.metatensor[source]
         #print(start)
         self.metatensor['cluster_id']=cluster_id
         for i in range(0,seq_length):
             if transformer_len is not None:
                 start_zeros = [zeros_append(transformer_len, len(start)) + start]
                 self.metatensor[func] = torch.tensor(np.array(start_zeros, dtype='int'), requires_grad=False)
                #  print(self.metatensor[func].shape)
             else:
                self.metatensor[func] = torch.tensor(np.array(start), requires_grad=False)
             if self.device is not None:
                self.metatensor[func] = self.metatensor[func].to(self.device)
             self.mode='generate'
#             print(self.metatensor[func])
        #  print(self.metatensor[func])
#             print("execute model")
             self.Run()
            #print(self.res)
            #del self.metatensor[func]
            #del self.metatensor['LSTM1']
            #  print(start)
            #  print(self.res_vector)
             if transformer_len is not None:
                #  print('start', start)
                #  print('resvec', self.res_vector)
                 start.append(self.res_vector[0])
                #  print('start', start)
                 result = start
             else:
                 if sampling == "maxprob":
                     start = [self.res_vector[len(start)-1]]
                     result.append(start[0])
                 if sampling == "random":
                    if random.randint(0,1)==0:
                        self.res_vector=self.second_result
                        start = [self.res_vector[len(start)-1]]
                        result.append(start[0])
                 if sampling == "nucleus":
                    q = nucleus_sampling()
                    start = [q]
                    result.append(q)
                
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
#               print(len(alphabet)+1)
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
#         if self.device is not None:
#           print("DEVICE", self.metatensor[source].get_device())
         #print(self.metatensor[source])
         self.metatensor[target] = m(self.metatensor[source])
        # print(self.metatensor[target])
        # print(target)
         if self.device is not None:
             self.metatensor[target] = self.metatensor[target].to(self.device)
         self.last_call = target
         self.res = self.metatensor[target]

        return self

    def PositionalEmbedding(self, source = None, target = 'PositionalEmbedding', size = None, max_seq_len = None):
        """Positional Embedding Layer

        Keyword Arguments:
            size -- embedding size (default: {None})
            max_seq_len  -- chunk size (default: {None})
        """
        if source is None:
            source = self.last_call

        if self.mode=="design":
            if size is None:
                size = self.metatensor[source].shape[2]
            if max_seq_len is None:
                max_seq_len = self.metatensor[source].shape[1]
            pe = torch.zeros(max_seq_len, size)
            for pos in range(max_seq_len):
                for i in range(0, size, 2):
                    pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/size)))
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/size)))

        # print(pe)
            self.register_name('pe', pe)
            self.record('PositionalEmbedding',['source', 'target','size', 'max_seq_len'],[source, target, size, max_seq_len])
            self.last_call = target
        if source in self.metatensor:
            pe = self.layers['pe']
            self.metatensor[target] = self.metatensor[source] + pe
            self.last_call = target
            self.res = self.metatensor[target]
        return self

    def MultiHeadAttention(self, source = None, target = None, size = 64, input_size = None, head_dim = 8):
        """Multi Head Attention Layer

        Keyword Arguments:
            size -- size of keys, values, queries and output matrix (default: {64})
            input_size  -- size of input or size of embedding (default: {None})
            head_dim -- attention heads dimension (default: {8})
        """
        if source is None:
            source = self.last_call

        if self.mode=="design":
            if input_size is None:
                input_size = self.metatensor[source].shape[2]

            m = nn.MultiheadAttention(input_size, head_dim, kdim = input_size, vdim = input_size)
            if target is None:
              target = self.get_name("MultiHeadAttention",m)
            else:
               self.register_name(target,m)
            self.record('MultiHeadAttention',['source', 'target','size','input_size', 'head_dim'],[source, target, size, input_size, head_dim])

            # self.Linear(size=size, input_size=input_size, source=source, target = 'Key')
            # self.Linear(size=size, input_size=input_size, source=source, target = 'Value')
            # self.Linear(size=size, input_size=input_size, source=source, target = 'Query')
            self.record('MultiHeadAttention',['source', 'target','size', 'input_size', 'head_dim'],[source, target, size, input_size, head_dim])
            self.last_call = target
        else:
            m = self.layers[target]
        if source in self.metatensor:
            attn_output, attn_output_weights = m(self.metatensor[source], self.metatensor[source], self.metatensor[source])
            self.metatensor[target] = attn_output
            self.last_call = target
            self.res = self.metatensor[target]
        return self

    def TransformerFeedForward(self, source = None, target = 'TransformerFeedForward', input_size = None, last = True):
        """Feed forward ONLY for transformer"""
        if source is None:
            source = self.last_call

        if self.mode=="design":
            if input_size is None:
                input_size = self.metatensor[source].shape[2]
            self.Linear(size=2048, input_size=input_size, source=source)
            self.RelU()
            self.Linear(size=input_size, input_size=2048, target = target)
            self.last_call = target
            self.record('TransformerFeedForward',['source', 'target', 'input_size', 'last'],[source, target, input_size, last])
        if source in self.metatensor:
            self.last_call = target
            if last:
                self.metatensor[target] = self.metatensor[self.last_call].contiguous().view(self.metatensor[self.last_call].size()[0], -1)
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
             #print(shape)
             if len(shape)==2:
                           #LSTM shape needs 3 dimensions, adding batch dim by defaule
                         #  print("conversion")
                           if self.psize is not None:
                            self.metatensor[source] = self.metatensor[source].view(self.psize)
                           else:
                            self.metatensor[source] = self.metatensor[source].unsqueeze(0)

                           self.psize= None
                           #q =  (self.metatensor[source].size()[0] * self.metatensor[source].size()[1]) * (self.batch_size * self.metatensor[source].size()[1])
                           #print(q)
                           #print(self.metatensor[source].size())
                           #self.metatensor[source] = self.metatensor[source].unsqueeze(0)

                           #self.metatensor[source] = self.metatensor[source].view(self.batch_size,self.metatensor[source].size()[2],q)

                           #print(shape(self.metatensor[source])
            else:
                print("Warning: no data in model found in LSTM Layer")

            print(self.metatensor[source].shape)
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
                m = SRU(input_size, size, bidirectional = False, num_layers=depth,has_skip_term=True)
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
                #m.flatten_parameters()
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

    def LSTMBank(self, size:int=25, source = None, index_size=10, target = None, name=None, state_vector="default", input_size=None,carry_state="no", data_parallel=False):
            '''research layer. Do not use unless you know what you are doing
               if it breaks, you have to keep both pieces'''

            def repackage_hidden(h):
              """ hidden states in new Tensors, to detach them from their history."""
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
                            if self.psize is not None:
                             self.metatensor[source] = self.metatensor[source].view(self.psize)
                            else:
                             self.metatensor[source] = self.metatensor[source].unsqueeze(0)

                            self.psize= None
                    
#               self.metatensor[source] = self.metatensor[source].unsqueeze(0)
                           #q =  (self.metatensor[source].size()[0] * self.metatensor[source].size()[1]) * (self.batch_size * self.metatensor[source].size()[1])
                           #print(q)
                           #print(self.metatensor[source].size())
                           #self.metatensor[source] = self.metatensor[source].unsqueeze(0)

                           #self.metatensor[source] = self.metatensor[source].view(self.batch_size,self.metatensor[source].size()[2],q)

                           #print(shape(self.metatensor[source])
            else:
                print("Warning: no data in model found in LSTM Layer")

#            if int(self.metatensor['cluster_id'])>7:
#                 self.metatensor['cluster_id']='7'

            if self.mode=="design":
               #print(dim_size(self.metatensor[source]))
               #print(self.metatensor[source].size())
               if input_size is None:
                   if not source in self.metatensor:
                       print("Error: No data in the model and no input_size specified")
                       return
                   input_size = dim_size(self.metatensor[source])
               if not (name in self.layers):
                name = target
                layers_bank = [nn.LSTM(input_size, size, bidirectional = False, batch_first=True) for i in range(0,int(index_size))]

            #    m = layers_bank[int(int(self.metatensor['cluster_id'])/1)]
                m = layers_bank[0]

               else:
                #m = self.layers[name+str(int(int(self.metatensor['cluster_id'])/1))]
                pass

               if target is None:
                   #target = self.get_name("LSTMBank",m)
                   #print(target)
                   target='LSTMBank'
                   for i,x in enumerate(layers_bank):
                      print(i)
                      self.register_name(target+str(i), x)


               if name is None:
                  name=target
               print(target)
               self.record("LSTMBank",['size','source','target','input_size','name','state_vector','carry_state','data_parallel','index_size'],[size,source,target,input_size,name,state_vector,carry_state,data_parallel,index_size])
            else:
                 #print(shape)
               #import random
               m = self.layers[name+str(int(int(self.metatensor['cluster_id'])))]
               #m = self.layers[name+'4']

#   m.flatten_parameters()
               if data_parallel and self.mode=='train' :
                    m = nn.DataParallel(m)

            if source in self.metatensor:
              if not target+'_state' in self.metatensor:
                if state_vector=="default":
                  if not next(m.parameters()).is_cuda and self.mode!='design':
                     self.all_bank_to_cpu()
                     m.to('cuda:0')
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
    def all_bank_to_cpu(self):
     #put all bank on cpu
     for i in range(1,10):
        self.layers['LSTMBank'+str(i)].to('cpu')


    def LSTM(self, size:int=25, source = None, target = None, name=None, state_vector="default", input_size=None,carry_state="no", data_parallel=False):
            '''lstm layer'''

            def repackage_hidden(h):
              """ hidden states in new Tensors, to detach them from their history."""
              if isinstance(h, torch.Tensor):
                return h.detach()
              else:
                return tuple(repackage_hidden(v) for v in h)

            if source is None:
                source = self.last_call

            if source in self.metatensor:
             shape = self.metatensor[source].size()
#             print(shape,source)
             m = None
             if len(shape)==2:
                           #LSTM shape needs 3 dimensions, adding batch dim by defaule
                         #  print("conversion")
                           self.metatensor[source] = self.metatensor[source].unsqueeze(0)
                           #q =  (self.metatensor[source].size()[0] * self.metatensor[source].size()[1]) * (self.batch_size * self.metatensor[source].size()[1])
                           #print(q)
                           #print(self.metatensor[source].size())
                           #self.metatensor[source] = self.metatensor[source].unsqueeze(0)

                           #self.metatensor[source] = self.metatensor[source].view(self.batch_size,self.metatensor[source].size()[2],q)

             #print(shape)
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
               if not (name in self.layers):
                name = target
                m = nn.LSTM(input_size, size, bidirectional = False, batch_first=True)

               else:
                m = self.layers[name]

               if target is None:
                   target = self.get_name("LSTM",m)
               else:
                   if target not in self.layers:
                    self.register_name(target, m)
               if name is None:
                  name=target
               print(target)
               self.record("LSTM",['size','source','target','input_size','name','state_vector','carry_state','data_parallel'],[str(size),source,target,input_size,name,state_vector,carry_state,data_parallel])
            else:
                 #print(shape)

               m = self.layers[name]
               #q.flatten_parameters()
               if data_parallel and self.mode=='train' :
                    m = nn.DataParallel(m)
            if m is None:
               m = self.layers[name]

            if source in self.metatensor:
              if not target+'_state' in self.metatensor:
                if state_vector=="default":
                  #if self.mode!='design':
                   #m.module.flatten_parameters()
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
              if self.mode=='generate' or self.mode=='predict':
                  self.metatensor[target+'_state'] = repackage_hidden(hidden)
        #         print("hidden stored")
#              self.res = self.metatensor[target]
            self.last_call = target
            return self


    def Regression(self, target=None,regression_target=None, source=None,input_size=None):
        global res
        if source is None:
            source = self.last_call

        #if len(self)>0:
        criterion = nn.MSELoss()
        if source in self.metatensor:
            if len(self.metatensor[source].shape)==4: #and (len(self.metatensor[regression_target].shape)==3):
                self.metatensor[source] = self.metatensor[source].squeeze(1)

        if self.mode=='design':
            self.record("Regression",['source','target','regression_target'],[source,target,regression_target])
            if source in self.metatensor:
              self.res = self.metatensor[source]

        if self.mode =='train' or self.mode == 'eval':
            maxsize = min(len(self),self.batch_size)
            self.res = self.metatensor[source]
            self.loss = criterion(self.res, self.metatensor[regression_target])
        if source in self.metatensor:
            self.metatensor[target]=self.res
        return self


    def Classify(self,target=None, class_target=None, cmode="Normal", source=None, input_size = None,data_parallel=False):
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
        #print("classifier input shape", self.metatensor[source].shape)
        if source in self.metatensor:
         if source in self.layers:
             prev_layer = self.layers[source]
             if sru_enabled:
              if (type(prev_layer) is torch.nn.LSTM) or  (type(prev_layer) is SRU):
                    self.metatensor[source] = self.metatensor[source].contiguous().view(unisize(self.metatensor[source]), -1)
                #    print("size conversion", self.metatensor[source].size())
             else:
              if (type(prev_layer) is torch.nn.LSTM):
                    self.metatensor[source] = self.metatensor[source].contiguous().view(unisize(self.metatensor[source]), -1)
        #if self.mode=='train':
        # criterion = nn.NLLLoss()
        #else:
        criterion = nn.CrossEntropyLoss()

        if self.mode=='design':
        #    print(class_target)

            self.class_target = class_target
            if len(self)>0 and len(self.classes)==0:
               self.classes = self.Distinct(class_target)

            if input_size is None:
               input_size = tin_size(self.metatensor[source])
        #    print(input_size)
            print(len(self.classes))
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
            if data_parallel and self.mode=='train':
                m=nn.DataParallel(m)
                s=nn.DataParllel(s)

            if len(self)>0:
             s = nn.Softmax(dim=-1)
             self.res = s(m(self.metatensor[source]))
            #print("Classification loss:" + str(self.loss))
            self.record("Classify",['target','class_target','cmode','source', 'input_size'],[target,class_target,cmode, source, input_size])
            print("modes", self.target, self.class_target, len(self))
            return self



        if self.mode =='train' or self.mode == 'eval':
            maxsize = min(len(self),self.batch_size)
#            print("target=",class_target)
            m = self.layers[target]
            if class_target!="_seq_model":
        #       print("CLAASS:",len(self[self.index:self.index+self.batch_size]))
               class_vector = self.vectorize_classes(self[self.index:self.index+self.batch_size], class_target)
        #       print(class_vector.shape)
            else:
               class_vector = self.metatensor['classes'].contiguous().view(unisize(self.metatensor[source]))
#               print("wrong_class")
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

#            print("**********************")
#            print(self.res.get_device())
#            print(class_vector.get_device())

            self.loss = criterion(self.res, class_vector)

            s=nn.Softmax(dim=-1)
        #    s = nn.LogSoftmax(dim=-1)
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
            try:
             for x in self.res:
                self[i][target] = x
    #           print(x)
                i = i + 1
            except:
                pass



        if self.mode == 'predict' or self.mode=='generate':
            m = self.layers[target]
            self.res = m(self.metatensor[source])

            s = nn.Softmax(dim=-1)

            self.res = s(self.res)
            self.metatensor[target]=self.res
            if self.device is None:
              nx = self.res.detach().numpy()
            else:
              nx = self.res.cpu().detach().numpy()
            #print(n)
            p = False
            if len(nx[0])>2:
#              print(n.shape)
#              print(n[-1])
              z = np.argsort(nx,axis=1)
              p = True
            #  print(n.shape)

              n1 = z[:,-2]
              n =  z[:,-1]
            else:
              n = list(np.argmax(nx, axis = 1))


            #print(self.classes)
#            print("here")
            #print(n)
            self.metatensor['class_probs'] = nx
            self.res_vector = n
            if p:
             self.second_result = n1
             
             self.metatensor['class_sorted'] = z
        #    print(n)
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
#                print(x,target,i)
                i = i + 1
            except:
                pass
#                print("error!")
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

    def Precision(self,target,prediction,term):
        mode = self.mode
        self.mode='predict'
        total_predicted = self.Match({prediction:term})
        total_predicted.mode='predict'
        correctly_predicted = total_predicted.Match({target:term})
        total_terms = self.Match({target:term})
        if len(total_predicted)>0:
            return len(correctly_predicted) /len(total_predicted)
        else:
            return 0.0

    def Recall(self,target,prediction,term):
        mode = self.mode
        self.mode='predict'
        total_predicted = self.Match({prediction:term})
        total_predicted.mode='predict'
        correctly_predicted = total_predicted.Match({target:term})
        total_terms = self.Match({target:term})
        if len(total_terms)>0:
            return len(correctly_predicted) /len(total_terms)
        else:
            return 0.0

    def F1(self,target,prediction,term):
        prec = self.Precision(target,prediction,term)
        rec = self.Recall(target,prediction,term)
        return (2 * prec * rec)/(prec+rec)

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

    def Exec(self,cache=True):

        if cache and len(self)>0:
            if not 'feature_tensor' in self[self.index]:
                self.Run()

        if self.Child[-1].mode=='design':
         self.Child[-1].record("Parent[0].Exec",['cache'],[cache])
        return self

    def AddParentModel(self, model):
        if self.Parent is None:
            self.Parent = []
        self.Parent.append(model)
        if model.Child is None:
            model.Child = []
        model.Child.append(self)

    def DistributeFeatures(self, source="", target="",cache=True):

        if len(self)>0:
         node = self[self.index]
         children = node.Children({})
        # print("D",self.metatensor)
         p = []
         if len(self.metatensor[target].shape)==3:
            self.metatensor[target] = self.metatensor[target].contiguous().view(unisize(self.metatensor[target]), -1)

         if cache:
             if 'feature_tensor'+target in node:
                pass
             else:

               for i,x in enumerate(node[source+'_inds_']):
                   p.append(self.metatensor[target][x].detach().cpu())

         if not cache:
               for i,x in enumerate(node[source+'_inds_']):
                   p.append(self.metatensor[target][x].detach().cpu())

         if cache:
              if not 'feature_tensor'+target in node:
                  node['feature_tensor'+target] = p
                  node['feature_tensor']=''
        #          print("cached p: ", len(p))
              else:
                  p = node['feature_tensor'+target]
#                  print("uncached p", len(p))

         #res = [x[target] for x in children]
         #print(self.device)
         if self.device is not None:
        #    print(p,target)
            for x in self.Child:
               x.metatensor[target] = torch.stack(p,dim=0).detach().to(self.device)
         else:
            for x in self.Child:
                x.metatensor[target] = torch.stack(p,dim=0).detach()
        # print("after distibute shape", self.Child[0].metatensor[target].shape)
        # print(target)
        # print(self.Child[0].title)

        if self.Child[-1].mode=='design':
           self.Child[-1].record('Parent[0].DistributeFeatures',['source','target','cache'],[source, target,cache])
       # else:
       #     self.next()
        return self

    def CopyState(self,source,target):
        state = self.metatensor[source]
        self.Child[0].metatensor[target] = state[-1].unsqueeze(0)
        if self.Child[0].mode=='design':
            self.Child[0].record('Parent[0].CopyState',['source','target'],[source, target])

    def LoadState(self,filename, strict=True):
        d =  torch.load(filename,map_location='cpu')
    #    pretrained_dict = {k: v for k, v in d.items() if  "Classify1" not in k}
        # 2. overwrite entries in the existing state dict
    #    self.state_dict().update(pretrained_dict)
        # 3. load the new state dict
        #model.load_state_dict(pretrained_dict)
        print(d.keys())
        self.load_state_dict(d, strict=False)

    def LoadStateBank(self, filename, bankcount, source_layer):
        d = torch.load(filename,map_location='cpu')
        self.load_state_dict(torch.load(filename,map_location='cpu'), strict=False)
        for i in range(0,bankcount):
            layer = self.layers['LSTMBank' + str(i)]
            dx = layer.state_dict()
            print(self.layers)
            w = d
            w1 = d[source_layer+'.'+'weight_ih_l0']
            print(dx.keys())
            dx['weight_ih_l0'].copy_(w1)
            dx['weight_hh_l0'].copy_(d[source_layer+'.'+'weight_hh_l0'])
            dx['bias_ih_l0'].copy_(d[source_layer+'.'+'bias_ih_l0'])
            dx['bias_hh_l0'].copy_(d[source_layer+'.'+'bias_hh_l0'])

    def Unroll(self, target_list=None):
        graph = self.parent_graph
        lst = dNodeList(graph) if target_list is None else target_list
        lst.save_func = self.save_func
        for x in self:
            words = x.Children({})
            for word in words:
                lst.append(word)
        lst.Parent = [self]
        if self.Child is not None:
          self.Child.append(lst)
        else:
          self.Child=[lst]
        self.batch_size = 1 #TODO: hack, to be removed
        return lst

    def ConvertTo(self, ctype:str="sentence"):
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
                 if x in allf:
                     ind = allf[x]
                     vector[ind] = 1.0


             nodes[i][target] = torch.from_numpy(vector).float()
        # print("NODES=",len(nodes))
         res = [x[target] for x in nodes]
         self.metatensor[target] = torch.stack(res,dim=0)
         if self.device is not None:
             self.metatensor[target] = self.metatensor[target].to(self.device)
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
#        print(len(self))
#        print(self.metatensor)
#        print(source[0] in metatensor)
        if source[0] in self.metatensor and len(self)>0:
         #execute merge if we have necessary data computed
       #  print(len(self.metatensor[source[0]].shape),len(self.metatensor[source[1]].shape),len(source))
            if len(self.metatensor[source[0]].shape)==2 and len(self.metatensor[source[1]].shape)==3 and len(source)==2:
                if self.mode=='design':
                    print("WARNING: 2d to 3d merge defaults to sequence merge now, 2d data will be padded")
                    print(self.metatensor[source[0]].unsqueeze(0).shape, self.metatensor[source[1]].shape)
                output_tensor = torch.cat([self.metatensor[source[0]].unsqueeze(0),self.metatensor[source[1]]],2)

            if len(self.metatensor[source[0]].shape)==2 and output_tensor is None:
                #merge 2d data (batch*data)
                input_tensors = [self.metatensor[x] for x in source]
                output_tensor = torch.cat(input_tensors,1)

            if len(self.metatensor[source[0]].shape)==3 and output_tensor is None:
                #merge 3d data (batch*word*word_content)
                input_tensors = [self.metatensor[x] for x in source]
                output_tensor = torch.cat(input_tensors,2)

            if len(self.metatensor[source[0]].shape)==4 and output_tensor is None:
                #merge 4d data (batch*channel*x*y) over channel (stacking)
                input_tensors = [self.metatensor[x] for x in source]
                output_tensor = torch.cat(input_tensors,1)


            if output_tensor is None:
               print("ERROR: Operands shape is currently not supported")
               return
            self.metatensor[target] = output_tensor
            if self.device is not None:
               self.metatensor[target].to(self.device)
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
                       print(text)
                       if 'а' in text or 'о' in text or 'и' in text or 'е' in text or 'в' in text.lower():
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
                   if not target in self[0]:
                       self.BLSTM(size=size,source=source,target=target)
                   else:
                       self.BLSTM(size=size,source=source).Sigmoid()
                       self.Classify(class_target=target,target='dMap1')
                       self.Rename(source='dMap1',target=target)
                   return self

    def dCopyField(self,source=None,target=None, level="Parent", source_index=None,target_index=0,crange=None):
        '''copy content of one field to another. can copy to current or to parent model'''
        if source is None:
           source=self.last_call
        if target is None:
            target = source

        if self.mode=='design':
             self.record('dCopyField',['source','target','level','source_index','target_index','crange'],[source,target,level,source_index,target_index,crange])


        if source_index is None:
            source_index =   self.index
            crange = self.batch_size


        if level=="Parent":
            copymodel = self.Parent[0]
        else:
            copymodel = self
        for i in range(0,crange):
            #if copy model does not have nodes in that range, create them
            if len(copymodel)<target_index+i+1:
                while(len(copymodel)<target_index+i+1):
                    copymodel.append(Node(copymodel.parent_graph,{}))
            copymodel[target_index+i][target] = self[source_index+i][source]

    def Vectorizenumpy(self,  target=None, source=None):
        '''Vectorizes image '''
        if source is None:
            source = self.last_call
        if target is None:
            target = "image_tensor"


        if self.mode=='design':
            #if mode is design, limit number of elements to process
            print("INFO: Design mode, vectorizing 50 elements to test")
            maxnodes = min(50,len(self))
            nodes = self[0:maxnodes]
            self.record("Vectorizenumpy",['source','target'],[source,target])

        if len(self)>0 and source in self[0]:
            if self.mode!='design':
                nodes = self[self.index:self.index+self.batch_size]

            if len(self)>0:
             for x in nodes:
                x[target] = torch.from_numpy(x[source]).float()

            res = [x[target] for x in nodes]
            self.metatensor[target] = torch.stack(res,dim=0)
            if self.device is not None:
               self.metatensor[target] = self.metatensor[target].to(self.device)
            self.res = self.metatensor[target]


            self.last_call = target
        return self

    def dMap(self, source:str, target:str, size:int=256):
        '''dMap is generic operation that corresponds to any differntiable module that transforms fixed size vector into another fixed size vector
           it is also automaticall vectorizes data types which it knows how to vectorize (image, word, sequence of words) but you can apply dMap after
           your own feature generation function
           Args:
            source: field from  data is taken
            target: field were to write result (also source of classes for training in design/compile time)
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
           if dtype is np.ndarray:
                print("Vec_nump")
                self.Vectorizenumpy(source=source, target=target)

                if not target in self[0]:
                   self.Linear(size=size, target=target).Sigmoid()
                else:
                   self.Classify(class_target=target,target='dMap1')
                   self.Rename(source='dMap1',target=target)
                return self

        print("dMap Error: Data type is not supported")
        return self







    #compilation routines
    def Run(self):
        '''executes all recordered model code'''
        exec(self.code)

    def next(self):
        '''this function moves model to the next batch'''
        if self.index + self.batch_size < len(self):#-(self.batch_size):
            self.index = self.index + self.batch_size
        else:
            if self.loadfunc is None:
                self.index = 0
            else:
                self.parent_graph = Graph()
                self.loadfunc()
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
        model2.title="test"
        #model2.classes = self.classes
        model2.index = 0
        model2.classes = self.classes
        model2.batch_size = self.batch_size
        #copy data content with split
        if split==1.0:
            print("no split")
            return model2
 
        if not empty:
         copy_nodes = self[int(len(self)*split):]
         if self.Parent is not None:
          en_copy=False
         else:
             en_copy=True

         for x in copy_nodes:
             if self.Parent is not None:
                 if len(x.Parents({}))>0:
                  real_c = x.Parents({})[0].Children({})
                  if real_c[0] == x:
                     en_copy=True
                 else:
                     en_copy=False
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
            self.ResetLSTMState()
            self.batch_size = 1
            if (self.Parent is not None) and len(self.Parent)>0:
                self.Parent[0].ResetLSTMState()
           # print('-----')
            self.index = 0
          #  self.result=None
            self.total_loss = 0.0
            if len(self)>1000:
             print(len(self),"size of test set")

            while self.index<len(self):
        #        print(self.index)
                self.mode = basemode
#                self.metatensor={}
                self.Run()
                if basemode=='train' or basemode=='eval':
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
                if len(self)>10:
                    print("predicting: "+str(self.index)+"                       ", end="\r", flush=True)
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

        finished = False
        while not finished:
            for x in data:
                new_list.append(x)

            if new_list.Parent is not None:
                parent = new_list.Parent[0]
                parent.clear()
                conv = data.ConvertTo()
                for x in conv:
                    parent.append(x)
                    parent.parent_graph = data.parent_graph
                    new_list.parent_graph = data.parent_graph

            new_list.mode='predict'
            self.mode='predict'
            new_list.Predict2(basemode='predict')
            finished=True
            if getattr(data, "loadfunc", None) is not None:
                if data.loadfunc is not None:
                    data.clear()
                    data.loadfunc()
                    if len(data)>0:
                        finished=False


        return data


    def compile(self,  opt="Adam", size=50, lr=0.01,avg_steps = None,thresold = 0.002, post_func = None, weight_decay=3e-4, control_mode='test', mixed_precision=False,data_parallel=False,test_data=None,split_proportion=0.8):
        _compile(self, opt, size, lr, avg_steps=avg_steps, thresold=thresold, post_func=post_func, weight_decay=weight_decay,control_mode=control_mode, mixed_precision=mixed_precision,data_parallel=data_parallel,test_data=test_data,split_proportion=split_proportion)
        return self.predict

    def Export(self, filename):
            if filename=="model":
                print("ERROR: model is reserved keyword, please choose another name")
                return
            if 'model' in filename:
                print("ERROR: Your data processing function should not be named 'model', please choose a meaningful name")
                return
            os.mkdir(filename)
            self.load_state_dict(torch.load("tempmodel.tmp"))
            torch.save(self.state_dict(), "./" + filename + "/" + filename + ".mod")
            code = self.code
            cbase = []
            cbase.append("from neuthink import metagraph as m")
           # cbase.append("from neuthink.textviews import SentenceView")
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


def test_decode_translate(model):
    t1 = "Острый гастрит"[::-1]
    data = model.Unfold_aux(start=model.Text.StringEncode('\n'+t1+'^^'), seq_length=100)
    data = model.Text.StringDecode(data)

    t2 = "Псориаз"[::-1]
    data1 = model.Unfold_aux(start=model.Text.StringEncode('\n'+t2+'^^'), seq_length=100)
    data1 = model.Text.StringDecode(data1)
    t3 = "Элитриптан"[::-1]
    data2 = model.Unfold_aux(start=model.Text.StringEncode('\n'+t3+'^^'), seq_length=100)
    data2 = model.Text.StringDecode(data2)
    t4= "Синдром запястного канала"[::-1]
    data3 = model.Unfold_aux(start=model.Text.StringEncode('\n' +t4+'^^'), seq_length=100)
    data3 = model.Text.StringDecode(data3)


    model.mode = 'train'
    return data +'ST\n'+data1+'\nST'+data2 +'\nST'+data3

def test_decode_para(model):
    t1 = "патент государственным органом власти по собственности; например, российской федерации органом  роспатент"
    data = model.Unfold_aux(start=model.Text.StringEncode('\n'+t1+'^'), seq_length=200)
    data = model.Text.StringDecode(data)

    t2 = "правовая охрана, которую предоставляет патент, защищает решение, а не задачу"
    data1 = model.Unfold_aux(start=model.Text.StringEncode('\n'+t2+'^'), seq_length=200)
    data1 = model.Text.StringDecode(data1)
    t3 = "в соответствии с п.1 ст. 1363 гк рф, срок исключительного права на изобретение, модель, образец и удостоверяющего право патента  со дня подачи заявки"
    data2 = model.Unfold_aux(start=model.Text.StringEncode('\n'+t3+'^'), seq_length=200)
    data2 = model.Text.StringDecode(data2)
    t4= "иностранные граждане, осуществившие въезд на территории РФ по режиму, могут устроиться на работу в 2015 году в России вступает в силу закон"
    data3 = model.Unfold_aux(start=model.Text.StringEncode('\n' +t4+'^'), seq_length=200)
    data3 = model.Text.StringDecode(data3)


    model.mode = 'train'
    return data +'ST\n'+data1+'\nST'+data2 +'\nST'+data3


def test_decode(model):
    data = model.Unfold_aux(start=model.Text.StringEncode('T'), seq_length=400)
    data = model.Text.StringDecode(data)
    model.mode='train'

    return data

def test_decode_ru(model):
    #model.disableGPU()
    data = model.Unfold_aux(start=model.Text.StringEncode('Это'), seq_length=420)
    data = model.Text.StringDecode(data)
    model.mode='train'
    #model.enableGPU('cuda:1')
    return data



def test_decode_transformer(model):
    data = model.Unfold_aux(start=model.Text.StringEncode('T'), seq_length=256, transformer_len = 300)
    data = model.Text.StringDecode(data)
    model.mode = 'train'
    return data

def test_decode_bank(model):
    data = model.Unfold_aux(start=model.Text.StringEncode('T'), seq_length=400,cluster_id=0)
    data = model.Text.StringDecode(data)
    data= data+'\n\n'
    datax = model.Unfold_aux(start=model.Text.StringEncode('T'), seq_length=400,cluster_id=2)
    data = data + model.Text.StringDecode(datax)
    data= data+'\n\n'
    datax = model.Unfold_aux(start=model.Text.StringEncode('T'), seq_length=400,cluster_id=4)
    data = data + model.Text.StringDecode(datax)

    model.mode = 'train'
    return data

def set_bank(model,optimizer,lr):
   #put all bank on cpu
   for i in range(1,10):
      model.layers['LSTMBank'+str(i)].to('cpu')
   #put current layer on GPU
   lindex = model[model.index]['cluster_id']
   model.layers['LSTMBank'+lindex].to('cuda:0')
   opt = torch.optim.SGD(model.parameters(), lr = lr ,momentum=0.8, nesterov=True)
   return opt




def _compile(model:dNodeList, opt="Adam", size=50, lr=0.01,avg_steps = None,thresold = 0.002, post_func=None, weight_decay=3e-4, control_mode="test",mixed_precision=False,data_parallel=False, test_data=None, split_proportion=0.8):


    error = 900000000000000
#    thresold = 0.002
    prev_error = 10000000
    best_f1 = ""
    test_f1 = ""
    accuracy = None
    best_accuracy = None
    moving_avg = 0
    best_loss = 900000000
    merror = 100000
#    avg_steps = 550
    counter = 0
    model.mode = "train"
    model.ResetLSTMState()
    if (model.Parent is not None) and len(model.Parent)>0:
         model.Parent[0].ResetLSTMState()
#    model.to('cuda:0')
    print(model.save_func)
    model.batch_size = size
    #orch.cuda.set_device(1)
#    opt='SGD'
    if opt=="SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = lr ,momentum=0.8, nesterov=True)
        #optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 3e-4)
    if opt=="Adam":
        #model.parameters().flatten_parameters()
        optimizer=  torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    #    optimizer2= torch.optim.Adam(parameters_advanced, lr = lr)# weight_decay = weight_decay)

        #optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    model.index = 0#185000#14 * 3500
    q = len(model)
    print("len of the model",q)
    print("split")
    if test_data is None:
        if len(model)<500000:
            test_model = model.Clone(split=split_proportion)
        else:
            if len(model)>1000000:
                print(1-(30000/len(model)))
                test_model = model.Clone(split=1-(15000/len(model)))
                print(len(test_model))
                print("Clone finished")
            else:
                test_model = model.Clone(split=0.9)
    else:
        test_model = model.Clone(split=1.0)
        for x in test_data:
            test_model.append(x)
        test_data.clear()
    model.test_set  = test_model
    random.seed(336334314)

    if avg_steps is None:
        avg_steps = len(model)/size

    import gc
    gc.collect()
    model.Shuffle()
    stats = open('stats.csv','a')
    stats.write('iteration \t train_error \t test_error \n')
    stats.close()
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,  loss_scale="dynamic")

    if data_parallel:
       test_model.batch_size = 21#int(size/torch.cuda.device_count())
#    if data_parallel:
#        torch.distributed.init_process_group(backend='nccl',init_method='env://')
#        model = DDP(model, delay_allreduce=True)

    step  = 0
    cind = 0
    while lr > thresold:
#        optimizer = set_bank(model,optimizer,lr)
        optimizer.zero_grad()
        model.Run()
        batch_error = model.loss
        torch.nn.utils.clip_grad_norm(model.parameters(),1)
        #batch_error = batch_error.mean()
#        v = batch_error.detach().item()

        if mixed_precision:
            with amp.scale_loss(batch_error, optimizer) as scaled_loss:
                scaled_loss.backward()

        else:
            batch_error.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        v = batch_error.detach().item()
        import math
        if  math.isnan(v):
          print(model.metatensor['cluster_id'])
        else:

           optimizer.step()
           optimizer.zero_grad()
#           cind = cind + 1
#           if cind > 700:
#             gc.collect()
#             cind=0
#           torch.cuda.empty_cache()

#        model.metatensor={}

        model.next()

        display_comp_status(counter, batch_error.detach().item(), error, test_f1, best_loss, best_f1, lr,test_accuracy=accuracy, best_accuracy = best_accuracy,train_moving_loss=merror)
        counter = counter + 1
        moving_avg = moving_avg + batch_error.detach().item()
        batch_error = 0

        if counter > avg_steps:
    #       if mixed_precision:
    #         with amp.scale_loss(batch_error, optimizer) as scaled_loss:
    #            print(scaled_loss.loss_scale)

           prev_error = error
           merror = moving_avg #/ avg_steps
#           if merror>1200:#
#             model.LoadState('tempmodel.tmp')
#             continue
           moving_avg = 0
           test_model.batch_size=21
#           all_to_gpu(model)
           torch.set_grad_enabled(False)
           test_model.Predict2(target=test_model.class_target, basemode='eval')
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
           error = test_model.total_loss if control_mode=='test' else merror
          # if data_parallel:
          #    rt = tensor.clone()
    #dist.all_reduce(rt, op=dist.reduce_op.SUM)
#rt /= args.world_size
           stats = open('stats.csv','a')

           stats.write(str(step) + '\t' + str(merror) + '\t' + str(error) + '\n' )
           stats.close()
           if best_loss > error:
#           if True:
               best_loss = error
               best_f1 = test_f1
               best_accuracy = accuracy
               torch.save(model.state_dict(), "tempmodel.tmp")


           if prev_error<error:
            lr = lr * 0.98
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
           torch.set_grad_enabled(True)
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
