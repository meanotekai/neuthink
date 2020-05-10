# -*- coding: utf-8 -*-

import types
import sys
from math import log
from typing import Generic, Mapping,Dict,TypeVar



def dextend(dic,field,unit):
  if field in dic:
      dic[field].append(unit)
  else:
      dic[field]=[unit]


class NodeList(list):
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
            return NodeList(self.parent_graph, list.__getitem__(self, given))
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

       return NodeList(self.parent_graph)

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

    def HasChild(self, node):
        result = []
        for x in self:
            people = x.children(node)
            if people:
                result.append(x)
        return result

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


    def Largest(self, feature):
        '''Find node with the lagrest value of specified property'''
        maxval = -100000
        curnode = {}
        for x in self:
            if feature in x:
                if float(x[feature]) > maxval:
                    maxval = float(x[feature])
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


    def Higher(self, feature, value):
        '''Find nodes with the property higher then value'''

        results = NodeList(self.parent_graph)
        for x in self:
            if feature in x:
                if float(x[feature]) > value:
                    results.append(x)

        return results

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

    def NotEmpty(self):
        return len(self) > 0

    def Empty(self):
        return len(self) == 0

    def Count(self):
        return len(self)

    def sum(self, node_property):
        # функция складывает значения всех полей свойства node_property
        # функция полезна, елси нужно посчитать количество чего-либо
        d = 0.0
        for x in self:
            if node_property in x:
                d += x[node_property]
        return d

    def GroupBy(self,field_name):
      #группирует NodeList по указанному полю
      field_name = (field_name)
      if ' ' in field_name:
          field_name = field_name.replace(' ', '_')
      tasks = {}
      for x in self:
          if field_name in x:
             dextend(tasks,field_name + ":" + x[field_name]  ,x)
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
        fields = [x for x in fields if x is not '_segment']
        fields = [x for x in fields if x is not 'id']
        fields = [x for x in fields if x is not 'type']
        fields = [x for x in fields if x is not 'имя']
        for i in range(len(fields)):
            if fields[i] == 'имя':
                fields[i] = ''
            if fields[i] =='_segment':
                fields[i] = ''
        fields = [x for x in fields if x is not '']
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
                self.dict = result[0].dict
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

    def HasChild(self, node):
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

    def NotEmpty(self):
        return self["type"] != "empty"
    
    def IsEmpty(self):
        return self["type"] == "empty"

    def Connect(self, target_node, label=None):
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

    def Children(self, cond={},label=""):
        '''Return all nodes that are connected to this node and
        satisfying condition <cond> that can be dictionary or bool function'''
        all_children = self.ConnectsTo(label)
        return all_children.Match(cond)


    def Parents(self, cond={}, label=""):
        '''Return all nodes that are connected to this node'''
        all_parents= self.ConnectedWith(label)
        return all_parents.Match(cond)


    def Child(self, condition, label=""):
        return self.Children(condition,label=label).First()


    def Parent(self, condition, label=""):
        return self.Parents(condition,label=label).First()


    def Inc(self,feature,amount):
        '''Increment value of specified node property with a given amount'''
        val = self[feature]
        self[feature] = float(val) + amount


    def Set(self,feature,amount):
        '''Increment value of specified node property with a given amount'''
        print()
        if (self['type']=='раздел' and feature =='статус' and amount == 'с проблемой') or (self['type']=='раздел' and feature =='решение') or (self['type']=='раздел' and feature =='проблема'):
            pass
        else:
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


    def Empty(self):
        return Node(self,{"type":"empty"})

    def DeleteNode(self,node):
        self.nodes.remove(node)
        if node in self.edges:
           del self.edges[node]
        for key in self.edges:
            for el in self.edges[key]:
                if el==node:
                    self.edges[key].remove(el)
        for pair in self.relations:
            if node in pair:
                del self.relations[pair]

    def DetachNode(self,node, label=""):
        for x in self.edges:
            if node in x:
               x.remove(node)

        if node in self.edges:
           del self.edges[node]

        for pair in self.relations:
            if label != "":
                if node in pair and self.relations[pair] == label:
                    del self.relations[pair]
            else:
                if node in pair:
                    del self.relations[pair]



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

    def MatchOne(self, node):
        return self.Match(node).First()

    def Largest(self, feature):
        '''Find node with the lagrest value of specified property'''
        maxval = -100000
        curnode = {}
        for x in self.nodes:
            if feature in x:
                if float(x[feature]) > maxval:
                    maxval = float(x[feature])
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
            match_list = NodeList(self.parent_graph)
            for x in self:
                match = node(x)
                if match:
                    match_list.append(x)
        return match_list
