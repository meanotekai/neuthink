'''metastruct implements node readers for structured content like SQL or XML'''

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
from neuthink import metagraph as m
import gzip


def LoadCSV(filename:str,separator:str=",") -> m.dNodeList:
    '''loads csv into nodelist using CSV reader'''
    import neuthink.metagraph as m

    graph = Graph()
    lines :  m.dNodeList = m.dNodeList(graph)
    f = open(filename)
    csv_reader = csv.reader(f,delimiter=separator)
    first_column = True
    try:
     for row in csv_reader:
        if first_column:
            column_names = row
         #   print(column_names)
            first_column = False
        else:
            _node = {column_names[i]:row[i] for i in range(len(row))}
            node = Node(graph,_node)
            lines.append(node)
    except:
      print("error occured while parsing csv")
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

def LoadPostgressSQLDumpTable(filename:str, tablename:str, maxlines=None, display=True, zipped=False, start_from=0):
    '''Loads SQL dump (postgress) into nodelist (loadds singe table specified by tablename)'''
    graph = Graph()
    import neuthink.metagraph as m
    lines :  m.dNodeList = m.dNodeList(graph)
    if zipped:
        f = gzip.open(filename, mode='rt')
    else:
        f = open(filename)

    state=0
    fields_list:List[str]=[]
    line_count =0
    for line in f:
        #locate table struct line
        if display:
            print("Lines loaded: " + str(len(lines))+"            ", end="\r", flush=True)

        if state==1: #we are reading table
#            if "COPY" in line: #newtable started
#                state=2
            values = line.split('\t')
            line_count = line_count + 1
            if line_count < start_from:
                continue
            if len(values) != len(fields_list):
                state=2 #table ended or corrputed
                print()

            _node = {fields_list[i].strip():values[i].strip() for i in range(len(values))}
            node = Node(graph,_node)
            lines.append(node)

        if maxlines is not None and len(lines)>maxlines:
            break
        if state==0 and tablename in line and (not '--' in line):
            fields_list = line[line.index('('):line.index(')')].split(',')
          #  print(len(fields_list))
            state=1
    print()
    return lines

def SavePostgressSQLDumpTable(nodes,filename:str, tablename:str, maxlines=None, display=True, zipped=False, append=False,column_list=[]):
     '''Loads SQL dump (postgress) into nodelist (loadds singe table specified by tablename)'''
     if zipped:
        if append:
            f = gzip.open(filename, mode='at')
        else:
            f = gzip.open(filename, mode='wt')
     else:
        if append:
            f = open(filename,"a")
        else:
            f = open(filename,"w")
     if not append:
        st = "COPY public."+tablename
        fields = column_list
        fstr = ",".join(fields)
        st = st + '(' + fstr + ')' +  'FROM stdin;'
        f.write(st+'\n')
     for x in nodes:
        f.write('\t'.join([x[field_name] for field_name in fields])+'\n')

     f.close()
