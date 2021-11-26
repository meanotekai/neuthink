# -*- coding: utf-8 -*-

import types
import sys
from math import log
from typing import Generic, Mapping,Dict,TypeVar, Optional, Union, Any
from neuthink import functional as fp
import csv


def dextend(dic,field,unit):
  if field in dic:
      dic[field].append(unit)
  else:
      dic[field]=[unit]


class NodeList(list):
    def __init__(self, parent_graph, nodes=None, *args):
        list.__init__(self, *args)
        self.parent_graph = parent_graph
        if nodes!=None:
           for x in nodes:
               if type(x)==dict:
                  q = Node(parent_graph,x)
               else:
                  q = x
               self.append(q)

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
        if type(given) is str:
            return [x[given] for x in self]
        if isinstance(given, slice):
            # do your handling for a slice object:
            return NodeList(self.parent_graph, list.__getitem__(self, given))
        else:
            # Do your handling for a plain index
            return list.__getitem__(self, given)

    def __setitem__(self, key, newvalue):
        if type(key) is str:
            for x in self:
                x[key] = newvalue
        else:
            return list.__setitem__(self, key, newvalue)

    def First(self) ->  'Node[str, Any]':
        if len(self)>0:
            return self[0]
        else:
            return self.parent_graph.Empty()

    def Delete(self):
       for x in self:
           x.Delete()

       return NodeList(self.parent_graph)

    def Set(self,field,value):
       for x in self:
           x.Set(field,value)

       return self
    
    def Get(self, field, default):
       if field in self:
          return self[field]

    def Detach(self,node):
       for x in self:
           x.Detach(node)

       return self

    def Children(self, cond)->'NodeList':
        '''nodes that are direct childrens of all nodes in nodelist
        satisfying condition cond'''
        result = NodeList(self.parent_graph)
        for x in self:
           result = result + x.Children(cond)
        return result

    def Parents(self, cond):
        '''nodes that are direct parents of all nodes in nodelist
        satisfying condition cond'''
        result = NodeList(self.parent_graph)
        for x in self:
           result  = result + x.Parents(cond)
        return result


    def children_all(self, cond={}):
        '''Return all nodes that are connected to this node and
        satisfying condition <cond> that can be dictionary or bool function'''
        if len(self) == 1:
            all_children = self[-1].ConnectsTo()
            return children_all.Match(cond)
        else:
            res = self.parent_graph.Match(cond)
            return res

    def AddParent(self,node):
       for x in self:
           x.AddParent(node)

       return self

    def HasChild(self, node):
        result = NodeList(self.parent_graph)
        for x in self:
            people = x.Children(node)
            if len(people)>0:
                result.append(x)
        return NodeList(self,result)

    def HasParent(self, node)->'NodeList':
        result = NodeList(self.parent_graph)
        for x in self:
            people = x.Parents(node)
            if len(people)>0:
                result.append(x)
        return NodeList(self,result)

    def Distinct(self, node_property):
        '''Returns list of distinct values of specified node property'''
        d = {}
        for x in self:
            if node_property in x:
                d[x[node_property]] = 1
        result = [x for x in d if x != '']
        return result
    
    def Values(self, node_property):
        '''Returns list values of the specified field for each node in the list'''
        result = [node[node_property] for node in self if node_property in node]
        return result

    def MatchOne(self,node):
        return self.Match(node).First()


    def Largest(self, feature):
        '''Find node with the lagrest value of specified property'''
        maxval = -100000000000
        curnode = self.parent_graph.Empty()
        for x in self:
            if feature in x:
                if float(x[feature]) > maxval:
                    maxval = float(x[feature])
                    curnode = x
        return curnode


    def Smallest(self, feature):
        '''Find node with the lagrest value of specified property'''
        minval = 1000000000000
        curnode = {}
        for x in self:
            if feature in x:
                if float(x[feature]) < minval:
                    minval = float(x[feature])
                    curnode = x
        return curnode

    def Smaller(self, feature, value):
        '''Find nodes with the property smaller then value'''

        results = NodeList(self.parent_graph)
        for x in self:
            if feature in x:
                if float(x[feature]) < value:
                    results.append(x)

        return results

    def Mapi(self, func,source,target):

        for i,x in enumerate(self):
            x[target] = func(x[source],i,self)
        return self

    def Map(self, func, source, target):
        for x in self:
             x[target] = func(x[source])
        return self

    def Higher(self, feature, value):
        '''Find nodes with the property higher then value'''

        results = NodeList(self.parent_graph)
        for x in self:
            if feature in x:
                if float(x[feature]) > value:
                    results.append(x)

        return results
        
    def Interval(self, feature, value_min, value_max):
        data = [x for x in self if (feature in x) and (float(x[feature]) > value_min and float(x[feature]) < value_max)]
        results = NodeList(self.parent_graph, data)
        return results

    def Match2Node2(self, node1, node2):
        """ Получает для каждого node1 список детей типа node2 """
        nodes1 = self.Match(node1)
        result = {}
        for node in nodes1:
            result[node] = node.Children(node2)
        return result

    def NotMatch(self,node):

        if type(node) is dict:
            match_list = NodeList(self.parent_graph)
            for x in self:
                match = self.__matchnode(node, x)
                if not match:
                    match_list.append(x)
        return match_list

    def Match(self,node):
        proc = False
        if type(node) is dict or type(node) is Node:
            match_list = NodeList(self.parent_graph)
            proc = True
            for x in self:
                match = self.__matchnode(node, x)
                if match:
                    match_list.append(x)

        if type(node) is types.FunctionType:
            match_list = NodeList(self.parent_graph)
            proc = True
            for x in self:
                match = node(x)
                if match:
                    match_list.append(x)
        if not proc:
            print("Error: unsupported type input for Match function", type(node))

        return match_list

    def NotEmpty(self)->bool:
        return len(self) > 0

    def Empty(self)->bool:
        return len(self) == 0

    def Count(self)->int:
        return len(self)

    def Sum(self, node_property:str)->float:
        # функция складывает значения всех полей свойства node_property
        # функция полезна, елси нужно посчитать количество чего-либо
        d = 0.0
        for x in self:
            if node_property in x:
                d += x[node_property]
        return d

    def GroupBy(self,field_name,add_field_name:bool=True):
      #группирует NodeList по указанному полю
      field_name = (field_name)
      if ' ' in field_name:
          field_name = field_name.replace(' ', '_')
      tasks = {}
      for x in self:
          if field_name in x:
            field = field_name + ":" + x[field_name] if add_field_name else x[field_name]
            dextend(tasks, field, x)
          else:
            dextend(tasks,"без значения",x)
      return tasks

    def Grouping_field(self):
        #вычисляет оптимальное поле для группировки
        #поле делит NodeList более равномерно
        coef = 0.0
        rs_field = ''
        if len(self) > 0:
            all_fields = self[0].dict
            fields = all_fields.keys()
        fields = list(fields)
        for x in self:
            new_x = x.dict
            new_x = new_x.keys()

            for y in new_x:
                if y not in fields:
                    fields.append(y)
            # for x in self:
            #     for y in fields:
            #         if y not in x:
            #             y = ''
            # fields = [x for x in fields if x is not '' ]
        fields = [x for x in fields if x != '_segment']
        fields = [x for x in fields if x != 'id']
        fields = [x for x in fields if x != 'type']
        fields = [x for x in fields if x != 'имя']
        for i in range(len(fields)):
            if fields[i] == 'имя':
                fields[i] = ''
            if fields[i] =='_segment':
                fields[i] = ''
        fields = [x for x in fields if x != '']
        for x in fields:
            result = self.GroupBy(x)
            res = result.values()
            res = [len(x) for x in res]
            k = 1.0
            kk = 0.0
            sum1 = 0.0
            for y in res:
                sum1 += y
            for y in res:
                #считаем энтропию
                kk = float(y)/sum1 * log(float(y)/sum1, 2)
                k *= kk
                k = k * (-1.0)
            if k > coef:
                coef = k
                rs_field = x
        return rs_field

    def Sorting(self, property):
        #сортирует NodeList по указанному параметру
        property = (property)
        if ' ' in property:
            property = property.replace(' ', '_')
#        print("**********************************************************")
#        print("start sorting")
#        print(property)
        for i in range(len(self), 0, -1):
            for j in range(1, i):
                #стандартный метод сортировки
                if property in self[j-1] and property in self[j]:
                    if float(self[j - 1][property]) < float(self[j][property]):
                        tmp = self[j - 1]
                        self[j - 1] = self[j]
                        self[j] = tmp
                if property in self[j] and property not in self[j-1]:
                    tmp = self[j - 1]
                    self[j - 1] = self[j]
                    self[j] = tmp
#        print(self)
#        print("end sorting")
        return self

    def GeneralizedSorting(self, property):
        #сортирует NodeList с учетом приоритетов разделов
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        section = []
        result = []
        for x in self:
            sec = x.parent({'type':'раздел'})
            section.append(sec)
        print("i found next sections:")
        section = list(tuple(section))
        section = NodeList(parent_graph = self.parent_graph)
        print(section)
        section = section.Sorting('приоритет')
        print('sorting sections:')
        print(section)
        for x in section:
            print('_______________')
            tasks = self.MatchOne({'раздел':x['раздел']})
            result.append(tasks.Sorting('приоритет'))
            print(x)
            print(tasks)
            print('_______________')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return result

    def HigherPriority(self):
        #возвращает из списка элемент с самым высоким приоритетом
        result = self.Sorting('приоритет')
        #print(result)
        return result[0]

    def AllHigherPriority(self):
        #получает у каждого пользователя невыполненную задачу с самым высоким приоритетом
        result = NodeList(parent_graph = self.parent_graph)
        users = self.parent_graph.Match({'type':'пользователь'})
        for x in users:
            try:
                higher_task = x.children({'type':'задача', 'статус':'не выполнен'}).HigherPriority()
                result.append(higher_task)
            except:
                print('у пользователя ' + str(x['имя']) + ' нет задач' )

        return result

    def Complete(self, feature, amount):
        '''complete amount to feature of node'''
        for i in range(len(self)):
            if feature in self[i]:
                val = self[i][feature]
                self[i][feature] = val + ' ' + amount
            else:
                self[i].Set(feature, amount)
        return self

    def GroupBySection(self):
        section = {}
        for ch in self:
            par_ch = ch.parent({'type':'раздел'})
            if par_ch['имя'] in section:
                section[par_ch['имя']].append(ch)
            else:
                section[par_ch['имя']] = [ch]
        return section

    def __add__(self,new):
        _list = NodeList(self.parent_graph)
        for x in self:
            _list.append(x)
        for x in new:
            _list.append(x)
        return _list


KT = TypeVar('KT')
VT = TypeVar('VT')

class Node(Mapping[KT,VT]):
    '''Node class represents a single graph node'''
    def __init__(self, parent_graph, dic:Dict[KT,VT], makenew=True,make_or_fetch=False):
        '''node constructor

        Args:
            dict - dictionary of node properties
            parent_graph - Graph object to add this node to'''

        self.dict = dic

        self.parent_graph = parent_graph

        if make_or_fetch:
            result = parent_graph.Match(dic)
            if len(result) > 0:
                self = result[0]
            else:
                node = parent_graph.AddNode(self)
                self.dict['id'] = node['id']

        if makenew and not make_or_fetch:
            node = parent_graph.AddNode(self)
            self.dict['id'] = node['id']

    def __iter__(self):
        return self.dict.__iter__()

    def __getitem__(self, item_name:KT)->VT:
        if (item_name == 'id'):
            return str(self.dict[item_name])
        return self.dict[item_name]

    def __setitem__(self, item_name, item_value):
        self.parent_graph.UpdateNodeProperty(self, item_name, item_value)
        self.dict[item_name] = item_value

    def __contains__(self, item):
        return item in self.dict

    def __str__(self):
         return str(self.dict)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.dict)

    #def __eq__(self):
    #    return self.dict.__eq__()

    def __hash__(self):
            return int(id(self)/16)

    def keys(self):
        return self.dict.keys()

    def Add(self, node):
        '''Add new node as child to this node (adds directional connection)

        Args:
            node - dictionary with properties or Node class or NodeList
        '''
        if type(node) is NodeList:
            for xnode in node:
                self.parent_graph.AddEdge(self, xnode)
            return node

        if type(node) is dict:
            node = Node(self.parent_graph, node)
            #self.parent_graph.AddNode(node)
        self.parent_graph.AddEdge(self, node)
        return node

    def HasChild(self, node)->bool:
        if node in self.Children(): return True
        else: return False

    def RelatedBy(self, stype, nodeend):
        res = False
        cons = nodeend.children({'type':stype})
        for x in cons:
            if len(x.children(self))>0:
                res = True
        if res == True:
            return False
        else:
            return self

    def CheckProperty(self, cond)->bool:
        #проверяет указанное свойство у объекта.
        result = []
        result = NodeList(self.parent_graph)
        result.append(self)
        result1 = result.Match(cond)
        if len(result1) > 0:
            return True
        else:
            return False


    def AddParent(self, node):
        '''Add new node as child to this node (adds directional connection)

        Args:
            node - dictionary with properties or Node class
        '''
        if type(node) is dict:
            node = Node(self.parent_graph, node)
            #self.parent_graph.AddNode(node)
        self.parent_graph.AddEdge(node, self)
        return self

    def MoveTo(self, node, source = None):
        current_user=''
        current_user2=''
        if type(node) is NodeList:
            for nod in node:
                self.MoveTo(nod, source)
            return self
        #print ("moveTO executed")
        #detach node from all parent nodes
        if len(node.Parents({}))>0:
            #print ("found parents")
            for x  in node.Parents({}):
                if source == None:
                    #print ("detach " + str(x) + " "+ str(node))
                    self.parent_graph.DetachNode(x, node)
                else:
                    if x['type']=='_создатель':
                        current_user=x.Parents({'type':'пользователь'})[0]
                    if x['type']=='пользователь':
                        current_user2=x
                    if source == x['type']:
                      #print ("detach " + str(x) + " "+ str(node))
                      self.parent_graph.DetachNode(x, node)
        ##connect node to new parent and return it

        return self.Add(node)

    def AddChain(self, node):
        '''Add new node as child to this node (adds directional connection)

        Args:
            node - dictionary with properties or Node class
        '''
        print('adddddddddddddddddd_____________________chain')
        if type(node) is NodeList:
            for xnode in node:
                self.parent_graph.AddEdge(self, xnode)
            return self

        if type(node) is dict:
            node = Node(self.parent_graph, node)
            #self.parent_graph.AddNode(node)
        self.parent_graph.AddEdge(self, node)
        return self

    def NotEmpty(self) -> bool:
        return self["type"] != "empty"

    def IsEmpty(self) -> bool:
        return self["type"] == "empty"

    def Connect(self, target_node:'Node[str, Any]', label:Optional[str]=None):
        '''Adds directed connection between this node and target node'''

        self.parent_graph.AddEdge(self, target_node, label)

    def Connect_branch(self, target_node,branch):
        self.parent_graph.MatchBranch(self, target_node,branch)

    def Connect2(self, target_node,ls_ansvers):
        '''Adds directed connection between this node and target node'''
        self.parent_graph.AddEdge2(self, target_node,ls_ansvers)

    def ConnectsTo(self,label=""):
        '''Return all nodes that are connected to this node'''
        return self.parent_graph.ConnectsTo(self,label)

    def ConnectedWith(self, label=""):
        '''Return all nodes that are connected to this node'''
        return self.parent_graph.ConnectedWith(self,label)

    def FirstConnected(self,label=""):
        '''Return first node connected to this node'''
        return self.parent_graph.ConnectsTo(self,label)[0]

    def Children(self, cond={},label="")->NodeList:
        '''Return all nodes that are connected to this node and
        satisfying condition <cond> that can be dictionary or bool function'''
        all_children = self.ConnectsTo(label)
        return all_children.Match(cond)


    def Parents(self, cond={}, label="")->NodeList:
        '''Return all nodes that are connected to this node'''
        all_parents= self.ConnectedWith(label)
        return all_parents.Match(cond)


    def Child(self, condition, label="")->'Node[str, Any]':
        return self.Children(condition,label=label).First()


    def Parent(self, condition, label="")->'Node[str, Any]':
        return self.Parents(condition,label=label).First()


    def Inc(self,feature,amount):
        '''Increment value of specified node property with a given amount'''
        val = self[feature]
        self[feature] = float(val) + amount


    def Set(self,feature,amount):
        '''Set value of specified node property with a given value'''
        # print()
        # if (self['type']=='раздел' and feature =='статус' and amount == 'с проблемой') or (self['type']=='раздел' and feature =='решение') or (self['type']=='раздел' and feature =='проблема'):
        #    pass
        #  else:
        self[feature] = amount

        return self


    def IncOrSet(self, feature, amount):
        '''Increment value of specified node property with a given amount,
        creates property if does not exist'''
        if feature in self:
            val = self[feature]
        else:
            val = 0.0
        self[feature] = float(val) + amount
        return self


    def Detach(self,node):
        self.parent_graph.DetachNode(node, self)
        return self


    def Complete(self, feature, amount):
        '''complete amount to feature of node'''
        val = self[feature]
        self[feature] = val + ' ' + amount
        return self


    def Delete(self):
         self.parent_graph.DeleteNode(self)
         return self


    def SuperChild(self, node):
        #функция проверяет наличие нода
        #если его нет то добавляет новый
        #print('Super Child')
        if not self.Children(node).NotEmpty():
            #print("branch if")
            self.Add(node)
            #print("noooooooooode")
            #print(node)
            return self.Children(node).First()
        else:
            #print(node)
            #print("branch else")
            return self.Children(node).First()

    def MainTask(self,node):
        st = []
        all_node = self.parent_graph.ConnectedWith(self)
        ch_all_node = all_node.Match({'type':'пользователь', 'имя':node['исполнитель']})

        for x in ch_all_node[0].children_all({'статус':'не выполнен'}):
            if 'приоритет' in x:
                st.append(float(x['приоритет']))
        #print(st)
        node.Set('приоритет',int(round(max(st)))+1)
        return self

    def SetRemove(self,new_type):
        self.parent_graph.SetRemove(self,new_type)
        return self


class Graph():

    def __init__(self):
        #super(Graph, self).__init__()
        self.nodes = []
        self.edges_parent = {}
        self.edges = {}
        self.relations = {}
        self.last_id = 0
        self.nodes.append(Node(self,{"type":"empty"}))

    def AllNodes(self)->NodeList:
        """Возвращает все узлы в графе"""
        return self.Match({})

    def NodesCount(self)->int:
        """Возвращает целочисленное количество узлов в графе"""
        return self.AllNodes().Count()

    def AddNode(self, node):
        if not "id" in node:
            node["id"] = str(self.last_id)
        self.last_id = self.last_id + 1
        self.nodes.append(node)
        return node

    def AddEdge(self, node1, node2, label:str=None):
       # print('ADDEDGE',node1, node2)
        if node1 in self.edges:
            self.edges[node1].append(node2)
        else:
            lst = NodeList(self)
            lst.append(node2)
            self.edges[node1] = lst


        if node2 in self.edges_parent:
            self.edges_parent[node2].append(node1)
        else:
            lst1 = NodeList(self)
            lst1.append(node1)
            self.edges_parent[node2] = lst1


        # Именование
        self.relations[node1,node2] = label



    def ConnectsTo(self,node, label:str=""):
        '''finds all nodes in the graph that are connected to selected node'''
        if node in self.edges:
            if label != "":
                return NodeList(self, [x for x in self.edges[node] if (node,x) in self.relations and self.relations[(node,x)] == label])
            else:
                return self.edges[node]
        else:
            return NodeList(self)



    def ConnectedWith(self, node, label:str=""):
        """finds the nodes with which the selected node is associated"""
        if node in self.edges_parent:
            if label != "":
                return NodeList(self, [x for x in self.edges_parent[node] if (x,node) in self.relations and self.relations[(x,node)] == label])
            else:
                return self.edges_parent[node]
        else:
            return NodeList(self)

    def Match2Node2(self, node1, node2):
        """ Получает для каждого node1 список детей типа node2 """
        nodes1 = self.Match(node1)
        result = {}
        for node in nodes1:
            result[node] = node.Children(node2)
        return result

    def Empty(self):
        return Node(self,{"type":"empty"})

    def _make_line(self, node):
        keys = node.keys
        line = "{"
        line = line + ','.join([x +': csvLine.' + x for x in keys]) + '}'
        return line

    def ExportNodeCSV(self, nodetype:str, filename:str=None)->str:
         '''Export one type of node to CSV file'''    

         filename = nodetype+'.csv' if filename is None else filename
         type_nodes = self.Match({'type':nodetype})
         keys = type_nodes[0].keys
         f = open(filename, 'w')
         writer = csv.DictWriter(f, fieldnames=keys)
         writer.writeheader()
         for row in type_nodes:
                writer.writerow(row._dict) 
         f.close()
         query = ('LOAD CSV WITH HEADERS FROM "file:///%s" AS csvLine CREATE (p:%s' % (filename, nodetype)) + self._make_line(type_nodes[0]) + ')'
         return query

    def ExportAllNodesCSV(self, filename:str)->None:
        '''Export nodes to CSV file'''
        
        all_types = self.Distinct('type')
        print(all_types)
        all_queries = []
        for node_type in all_types:
           all_queries.append(self.ExportNodeCSV(node_type))
        
        f = open(filename+ '_edges.csv', 'w')
        writer = csv.writer(f)
        writer.writerow(['id1', 'id2','type1','type2'])
               
        for node1 in self.edges:
            for node2 in node1:
                writer.writerow([node1['id'],node2['id'],node1['type'], node2['type']])
        f.close()
        query = 'LOAD CSV WITH HEADERS FROM "file:///%s" AS csvLine MATCH (node1:csvLine.type1 {id: toInteger(csvLine.id1)}),(node2:csvLine.type2 {id: toInteger(csvLine.id2)}) CREATE (node1)-->(node2)' % filename+ '_edges.csv'
        all_queries.append(query)
        fp.save_lines( filename, all_queries)

               
               






    def DeleteNode(self,node):
        self.nodes.remove(node)
        if node in self.edges:
           del self.edges[node]

        for key in self.edges:
            for el in self.edges[key]:
                if el==node:
                    self.edges[key].remove(el)
        dnodes =[]
        for pair in self.relations:
            if node in pair:
                dnodes.append(pair)
        for x in dnodes:
            self.relations.pop(x)

    def DetachNode(self,node, label=""):
        for x in self.edges:
            if node in x:
               x.remove(node)

        if node in self.edges:
           del self.edges[node]
        f = None
        for pair in self.relations:
            if label != "":
                if node in pair and self.relations[pair] == label:
                    f  = pair
            else:
                if node in pair:
                    f = pair
        if f is not None:
           self.relations.pop(f)




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

    def UpdateNodeProperty(self, node, property_name, new_property_value):
        '''Updates value of node property'''
        match_list = NodeList(self)
        if type(node) is dict:
            for x in self.nodes:
                match = self.__matchnode(node, x)
                if match:
                    match_list.append(x)
        for node_l in match_list:
            node_l[property_name] = new_property_value

        return match_list


    def Distinct(self, node_property):
        '''Returns list of distinct values of specified node property'''
        d = {}
        for x in self.nodes:
            if node_property in x:
                d[x[node_property]] = 1
        result = [x for x in d]
        return result
    

    def Values(self, node_property):
        '''Returns list values of the specified field for each node in the list'''
        result = [node[node_property] for node in self.nodes if node_property in node]
        return result


    def MatchOne(self, node: Union[Node[str, Any], Dict[str, Any]]) -> Node[str, Any]:
        return self.Match(node).First()

    def Largest(self, node, feature):
        '''Find node with the lagrest value of specified property'''
        maxval = -100000
        curnode = self.Empty()
        for x in self.nodes:
            if feature in x:
                if float(x[feature]) > maxval:
                    maxval = float(x[feature])
                    curnode = x
        return curnode

    def Smallest(self, feature):
        '''Find node with the smallest value of specified property'''
        minval = 100000
        curnode = {}
        for x in self.nodes:
            if feature in x:
                if float(x[feature]) < minval:
                    minval = float(x[feature])
                    curnode = x
        return curnode

    def Match(self, node):
        match_list = NodeList(self)
        if type(node) is dict:
            for x in self.nodes:
                match = self.__matchnode(node, x)
                if match:
                    match_list.append(x)

        if type(node) is types.FunctionType:
            match_list = NodeList(self)
            for x in self.nodes:
                match = node(x)
                if match:
                    match_list.append(x)
        return match_list
