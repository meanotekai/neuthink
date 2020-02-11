'''metacluster implements clustering functions'''

from typing import List,Tuple,Optional
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
import os
import pickle

import gzip
try:
    from sklearn.cluster import KMeans
    sklearn_aval = True
except:
    sklearn_aval = False

class MetaCluster(object):
    """Image class, as part of metamodel"""
    def __init__(self, ParentModel):

        super(MetaCluster, self).__init__()
        self.model = ParentModel
        self.graph = ParentModel.parent_graph
    def KMeans(self, n_clusters,source='vectors',target='cluster_id',maxdata=None,n_init=2):
        '''computes K-means clustering for dataset'''
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
        if maxdata is None:
            data = [x[source] for x in self.model]
        if sklearn_aval:
            self.vecs = None
            self.kmeans = KMeans()
        

    def KMeans(self, n_clusters:int,source:str='vectors',target:str='cluster_id',maxdata:Optional[int]=None, n_init=2):
        '''computes K-means clustering for dataset'''
        if sklearn_aval:
            self.kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
            if maxdata is None:
                data = [x[source] for x in self.model]
            else:
                data = [x[source] for x in self.model[0:maxdata]]
            
            self.kmeans.fit(data)
            #
            for i,x in enumerate(self.kmeans.labels_):
                self.model[i][target] = x
        else:
            print("clustering requires sckit learn to be installed")
        
        self.kmeans.fit(data)
        #
        for i,x in enumerate(self.kmeans.labels_):
            self.model[i][target] = x
    
    def GetClusters(self, data, target='cluster_id'):
        self.model.clear()
        for x in data:
            self.append(x)
        data = [x['vectors'] for x in self.model]
        labels = self.kmeans.predict(data)
    def GetClusters(self, data, source='vectors'):
        self.model.clear()
        for x in data:
            self.append(x)
        data = [x[source] for x in self.model]
        labels = kmeans.predict(data)
        for i,x in enumerate(labels):
            self.model[i][target] = x 

    def GetClustersInPlace(self,target='cluster_id'):

        data = [x['vectors'] for x in self.model]
        labels = self.kmeans.predict(data)
        for i,x in enumerate(labels):
            self.model[i][target] = x 

    
    def Export(self, filename:str)->None:
         os.mkdir(filename)
         cbase = []
         cbase.append("from neuthink import metagraph as m")
         cbase.append("from neuthink.graph.basics import Graph")
         cbase.append("from sklearn.cluster import KMeans")
         cbase.append("import os")
         cbase.append("graph = Graph()")
            #cbase.append("import torch")
         cbase.append("path = os.path.dirname(os.path.abspath(__file__))")
         cbase.append("print(path)")
         cbase.append("model = m.dNodeList(graph)")
         cbase.append("model.Cluster.kmeans.set_params(pickle.load('" + path + filename + ".mod')")
         cbase.append(filename + "= lambda x: model.GetClusters(x)") 
        