# -*- coding: utf-8 -*-
# Copyright 2016 Meanotek AI

'''
This module for various text-related dataviews
'''
import random
import json
import os
from mindy.graph import basics as graph
from .nlptools.entitys import entity_parser

from .text_helper import get_sentences
from .text_helper import load_words_save_separators
from .text_helper import open_text
from .text_helper import clear_text

def split(string ,delimeter):
    '''this is helper function to avoid crazy pythonish behavior of split'''

    s = string.split(delimeter)
    s = list(filter(lambda x: len(x)>0, s))
    return s

def save_json(obj, fname):
    f = open(fname, "w")
    jd = json.dumps(obj, indent = 1, ensure_ascii=False)
    f.write(jd)
    f.close()

def save_file(list_lines, fname):
    f = open(fname, "w")
    f.writelines(list_lines)
    f.close()

def save_file_st(st, fname):
    f = open(fname, "w")
    f.write(st)
    f.close()

class BaseView:
    '''Abstract class, representing any data view'''
    def __init__(self, config=None):
        '''initializes data view
        Args:
          config: dictionary of parameters for creating persistent graph
        '''
        if config is None:
            self.graph = graph.Graph()
        else:
            raise NotImplementedError

class TextView(BaseView):
    '''a view representing text as a whole, suitable for classification tasks'''
    def __init__(self, config=None):
        BaseView.__init__(self,config=config)
        self.texts = graph.NodeList(self.graph)


    def save_colums(self,filename):
        f = open(filename,'w')
        for x in self.texts:
            text = x['text']
            classt = x['target_class#p']
            classa = x['target_class']
            f.write(text + '\n' + classt + ' ' + classa + '\n\n' )
        f.close()



    def load_column(self, filename:str, text_ind:int, class_ind:int,  separator=" ",  maxlen = 1000, ) -> None:
        '''loads text view from file with columns file

        Args:
            text_ind - column index of text
            class_ind - colum index of text class'''
        f = open(filename, "r")
        cur_text = graph.Node(self.graph, {})
        i = 0
        for line in f:
            line = line.strip()
            spl = split(line, separator)
            i = i + 1
        #    print (i)
        #    print (spl)
         #   if len(spl)>class_ind+1:
          #      print (str(i) + "too much columns on this indexs")
           #     print (line)
            #    print(spl[class_ind])
            if len(spl) > class_ind and len(spl) <= class_ind + 1:
                cur_text["text"] = spl[text_ind]
                cur_text["target_class"] = spl[class_ind].lower()
                self.texts.append(cur_text)
                cur_text = graph.Node(self.graph, {})
            else:
                pass
           #     print ("Error on line " + str(i) + " not enough colums for reading")


    def to_data_source(self):
        source = DataSource()
        for text in self.texts:
            text_str = text["text"]
            text_class = text["target_class"]
            dic = {}
            dic["source"] = text_str
            dic["original"] = text
            dic["target_class"] = text["target_class"]
            source.AddEntry(dic)
        all_classes = self.graph.Distinct("target_class")
    #    print (all_classes)
        source.classes = all_classes
        source.aspect_name = "target_class"
        return source

    def shuffle(self):
        random.shuffle(self.texts)

    def splits(self, prop):
        self.shuffle()
        part1 = TextView()
        part2 = TextView()

        index = int(len(self.texts) * prop)
        part1.texts = self.texts[0:index]
        part1.graph = self.graph
        part2.texts = self.texts[index:]
        part2.graph = self.graph
        return part1, part2

    def class_accuracy(self, aspect_name:str) -> float:
        '''computes accuracy over certain aspect'''
        count = 0.0
        total = 0.0
        for text in self.texts:

            if "target_class#p"  in text:
                    if text["target_class#p"] == text["target_class"]:
                        count = count + 1
            total = total + 1
        return count / total

#~ #~
class EntityView(BaseView):
    def __init__(self, config=None):
        BaseView.__init__(self,config=config)
        self.entities = []
        self.graph = None

    def save_ison(self, fsave):
        li_dics = []
        for node in self.entities:
            li_dics.append(node.dict)
        save_json(li_dics, fsave)

    def save_ann(self, fname):
        i = 1
        li = []
        for node in self.entities:
            dic_node = node.dict
            text = dic_node["text"]
            pstart = int(dic_node["pos_start"])
            pend = int(dic_node["pos_end"])
            clas = dic_node["entity_subtype"]
            st = "T%d\t%s %d %d\t%s\n" % (i, clas, pstart, pend, text)
          #  print(st)
            li.append(st)
            i += 1
        save_file(li, fname)





class SentenceView(BaseView):
    '''a view representing text as a list of sentences with words'''
    def __init__(self, config=None):
        BaseView.__init__(self, config=config)
        self.sentences = graph.NodeList(self.graph)
        self.extended = False
        self.clean_text = ""

    def load_sentences(self,sentences):
        self.sentences = sentences
        for sentence in sentences:
          for word in sentence["words"]:
              graph.Node(self.graph,word)

    def to_entity_view(self, aspect="termtag#p"):
        entity_parser(self, "termtag#p")
        d = EntityView()
        d.graph = self.graph
        d.entities = self.graph.Match({"type":"entity"})
        return d

    #~ if need open text use function text = open_text(filename)
    def get_clean_text(self, text):
        """
        Очистка текста от можественных пробелов, символов табуляции,
        а так же от переноса строки.
        """
        self.clean_text = clear_text(text)

    def get_text(self, text):
        words = load_words_save_separators(text)

        wsents = get_sentences(words, text)

        self.columns = ["type", "pos_start", "pos_end", "word"]
        for sent in wsents:
            cur_sentence = graph.Node(self.graph, {})
            cur_sentence["words"] = graph.NodeList(self.graph)
            cur_sentence["type"] = "sentence"
            for word_i in sent:
                word = graph.Node(self.graph, {})
                word["type"] = "word"
                word["word"] = word_i["word"]
                word["pos_start"] = word_i["pos_start"]
                word["pos_end"] = word_i["pos_end"]
                cur_sentence["words"].append(word)
            self.sentences.append(cur_sentence)

    def save_clear_text(self, fname):
        save_file_st(self.clean_text, fname)

    def load_ann(self, filename:str):
        """
        Загрузка ann файла
        """
        file_ann = filename + ".ann"
        file_txt = filename + ".txt"
        textf = open(file_txt,"r")
        text = textf.read()
        textf.close()
        self.get_text(text)
        annf = open(file_ann, "r")
        ann = annf.readlines()
        annf.close()
        anns =[]
        for line in ann:
            b = line.split("\t")
            if len(b) > 2:
                bca = b[1].split(" ")
                bc = []
                #print (bca)
                for bb in bca[1:]:
                    bc.extend(bb.split(";"))
               # print (bc)
                for i in range(0,len(bc),2):
                    dic = {}
                    dic["type"] = bca[0]
                    dic["diap"] = (int(bc[i]), int(bc[i+1]))
                    #~ dic["text"] = b[2]
                    anns.append(dic)
        #~ print (anns)
        for sent_obj in self.sentences:
            sent = sent_obj["words"]
            for word in sent:
                #~ rr = 0
                for an in anns:
                    if int(word["pos_start"]) >= an["diap"][0] and int(word["pos_end"]) <= an["diap"][1]:
                        word["arm_type"] = an["type"]
                        #~ rr = 1
            #~ if





    def load_column(self, filename: str, separator=" ", maxlen=1000, default_class=None, expand=False, maxwords=None) -> None:

        '''loads sentence view from file with columns file'''
        f = open(filename, "r")
        column_names = split(f.readline().strip(), separator)
        cur_sentence = graph.Node(self.graph, {})
        cur_sentence["words"] = graph.NodeList(self.graph)
        self.columns = column_names
        line_num = 0
        self.extended = expand
        for line in f:
            line = line.strip()
            line = line.replace("\ufeff","")
            line_num  = line_num  + 1
            if (maxwords is not None) and (line_num>maxwords):
                return
            if "<STOP>" in line or line == "" or len(cur_sentence["words"])>maxlen:
                if len(cur_sentence["words"]) > 0:
                    self.sentences.append(cur_sentence)
                    cur_sentence = graph.Node(self.graph, {})
                    cur_sentence["words"] = graph.NodeList(self.graph)
                    cur_sentence["type"] = "sentence"
            else:
                word = graph.Node(self.graph, {})
                words = split(line, separator)
        #        print (len(words))
         #       print (len(column_names))
                #TODO: This code is horrible: need to simplify
                #normal situation
                if len(words) == len(column_names):
                    for i in range(0, len(column_names)):
                        word[column_names[i]] = words[i]
                        if expand == True:
                            nn = graph.Node(
                                self.graph,{"type":column_names[i],"value":words[i]})
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
                               nn=  graph.Node(
                                    self.graph,{"type":column_names[i], "value":words[i]})
                               word.Connect(nn)
                        #fill unspecified colums with default value
                    #    print (range(len(words), len(self.columns)))
                     #   print (words)
                        for i in range(len(words), len(self.columns)):
                            word[column_names[i]] = default_class
                            if expand is True:
                                nn = graph.Node(
                                    self.graph,{"type":column_names[i], "value":default_class})
                                word.Connect(nn)

                if len(words) > len(column_names):
                     #we have too many classes
                     #fill normal range
                     for i in range(0, len(column_names)):
                         word[column_names[i]] = words[i]
                         if expand == True:
                             nn = graph.Node(
                                 self.graph,{"type":column_names[i], "value":words[i]})
                             word.Connect(nn)
                     #if expand = true, fill load other classes into graph

                     for i in range(len(column_names) , len(words)):
                          if expand == True:
                            nn = graph.Node(
                                self.graph,{"type":column_names[-1],"value":words[i]})
                            word.Connect(nn)
                   #  print (word.children({"type":column_names[-1]}).Distinct("value"))

                word["type"] = "word"

                cur_sentence["words"].append(word)
        if len(cur_sentence["words"]) > 0:
            self.sentences.append(cur_sentence)

    def make_column_view(self, new_col_name, aspect_name=None, classes_list=[], default_class="other"):
        '''creates new column '''
        i = 0
        for x in self.sentences:
            for word in x["words"]:
                word[new_col_name] = ""
                i = i + 1
                if aspect_name is not None:
                    aspect_classes = word.children(
                        {"type": aspect_name}).Match(
                            lambda x: x["value"] in classes_list)
                    if len(aspect_classes) == 0:
                        word[new_col_name] = default_class
                    else:
                        word[new_col_name] = aspect_classes[0]["value"]
                    if len(aspect_classes) > 1:
                        print ("Ambigous slice " + str(aspect_classes.Distinct("value"))) #+ word["N2"])


    def save_column(self, filename:str, separator=" ", column_list=[],
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
        if not self.extended:
            for sentence in self.sentences[ofs:lim]:
                words = sentence["words"]
                for word in words:
                    save_str = ""
                    if word["word"].strip() != "":
                        for cl in column_list:
                            if cl in word:
                                save_str = save_str + str(word[cl]) + "  "
                            else:
                                save_str = save_str + "other" + "  "
                        f.write(save_str + "\n")
                f.write("<STOP> \n")
        else:
            print ("saving extended model")
            for sentence in self.sentences:
                words = sentence["words"]
                for word in words:
                    save_str = ""
                    for cl in column_list[0:-1]:
                        save_str = save_str + word[cl] + "  "
                  #  print (column_list[-1])
                    class_list = word.children({"type":column_list[-1]}).Distinct("value")
                    class_list = " ".join(class_list)
                  #  print (class_list)
                    save_str = save_str + class_list
                    f.write(save_str + "\n")
                f.write("<STOP> \n")
        f.close()

    def __getitem__(self, given):
        #this enables to generate sub text views
        if isinstance(given, slice):
            sentences = self.sentences[given]
            nw = SentenceView()
            nw.graph = self.graph
            nw.sentences = sentences
            nw.columns = self.columns
            nw.extended = self.extended
            return nw

        else:
            return self.sentences[given]

    def shuffle(self):
        random.shuffle(self.sentences)

    def splits(self, prop):
        self.shuffle()
        part1 = SentenceView()
        part2 = SentenceView()

        index = int(len(self.sentences) * prop)
        part1.sentences = self.sentences[0:index]
        part1.graph = self.graph
        part1.columns = self.columns
        part2.sentences = self.sentences[index:]
        part2.graph = self.graph
        part2.columns = self.columns
        return part1, part2


    # def to_data_source(self, words_name:str, aspect_name:str,
    #                 ofs:int=None, lim:int=None, istrain=True) -> DataSource:
    #     '''Converts content to typical sequence tagger datasource
    #        Args:
    #            words_name - name of attributte, with source data, usually words
    #            aspect_name - name of aspect to predict/train
    #            lim: limit of sentences to save
    #            ofs: offset sentences to save
    #     '''
    #     #ISSUE: offset and limit will be removed from this function
    #     #Reason: recommended way to do this is to create another view, make
    #     #subset of data and then save it
    #     if ofs is not None or lim is not None:
    #         print ("ofs and lim parameters are deprecated, please update your code")
    #
    #
    #     source = DataSource()
    #     for sentence in self.sentences[ofs:lim]:
    #         words = sentence["words"]
    #         dic = {}
    #         dic["source"] = [x[words_name] for x in words]
    #         dic["original"] = sentence["words"]
    #         if not self.extended:
    #             #single class
    #             if not (aspect_name in sentence["words"][0]):
    #                  if istrain:
    #                      print ("Missing required field " + aspect_name)
    #                      return None
    #                  else:
    #                     dic["target_class"] = ["other" for x in sentence["words"]]
    #             else:
    #                 dic["target_class"] = [x[aspect_name] for x in sentence["words"]]
    #         else:
    #             #multi class
    #             class_list = [x.children({"type":aspect_name}).Distinct("value") for x in sentence["words"]]
    #         source.AddEntry(dic)
    #     all_classes = self.graph.Distinct(aspect_name)
    #     all_classes.sort()
    # #    print (all_classes)
    #     source.classes = all_classes
    #     source.aspect_name = aspect_name
    #     return source

    def aspect_accuracy(self, aspect_name: str) -> float:
        '''computes accuracy over certain aspect'''
        pname = aspect_name + "#p"
        total_words = self.graph.Match({"type": "word"})
        correct_words = total_words.Match(
            lambda x: pname in x and x[pname] == x[aspect_name])
        return (correct_words.Count() / total_words.Count())


    def class_precision(self, aspect_name, class_name:str) -> dict:
        """
        Calculates precision for given aspect and class
        precision = predicted positives (PP) / true positives (TP)
        Args:
            aspect_name: aspect name to measure

        Returns: dictionary with measure for each class in aspect name
        """
        pname = aspect_name + "#p"
        #predicted value = class_name
        PP = self.graph.Match({"type": "word", pname: class_name})
        TP = PP.Match({aspect_name: class_name})
        if PP.Count() > 0:
            return (TP.Count() / PP.Count())
        else:
            return 0.0

    def class_recall(self, aspect_name, class_name: str) -> dict:
        """
        Calculates recall  for given aspect and class
        recall = predicted_positives (PP) / all_positives(AP)
        Args:
            aspect_name: aspect name
            class_name: name of the class

        Returns: dictionary with measure for each class in aspect name
        """
        pname = aspect_name + "#p"
        #predicted value = class_name
        PP = self.graph.Match({"type": "word", pname: class_name})
        AP = self.graph.Match({aspect_name: class_name})
        PP = AP.Match({"type": "word", pname: class_name})
        return (PP.Count() / AP.Count())

    def class_f1(self, aspect_name, class_name:str) -> float:
        """Calculates F1 measure for given aspect and class"""
        P = self.class_precision(aspect_name, class_name)
        R = self.class_recall(aspect_name, class_name)
        if (P + R) > 0:
            f1 = 2 * P * R / (P + R)
        else:
            f1 = None
        return f1

    def aspect_f1(self, aspect_name) -> float:
        '''Calculates F1 for all classes in a given aspect
        Args:
            aspect_name: aspect name to measure

        Returns: dictionary with measure for each class in aspect name'''
        classes = self.graph.Distinct(aspect_name)
        return [(x, self.class_f1(aspect_name, x)) for x in classes]


    def load_bd(self, filename: str) -> None:
        """
        Loads sentence view from big_dict structure file
        Args:
            filename: file name <str>
        """
        with open(filename) as f:
            data = json.load(f)
            column_names = ["word", "pos_start", "pos_end", "type", "tag",
                            "class", "tone"]
            self.columns = column_names
            for item in data:
                text = item['constant']['text']
                for i in range(0, len(text)):
                    sentence = text[i]
                    cur_sentence = graph.Node(self.graph, {})
                    cur_sentence["type"] = "sentence"
                    cur_sentence["words"] = graph.NodeList(self.graph)
                    for j in range(0, len(sentence)):
                        word = graph.Node(self.graph, {})
                        for c_name in column_names:
                            if c_name in ["class", "tone"]:
                                word[c_name] = sentence[j]["variable"][c_name][0]
                            elif c_name in ["pos_start", "pos_end"]:
                                word["pos_start"] = str(sentence[j]["pos"][0])
                                word["pos_end"] = str(sentence[j]["pos"][1])
                            else:
                                word[c_name] = sentence[j][c_name]
                        cur_sentence["words"].append(word)
                    self.sentences.append(cur_sentence)


    def save_ann(self, column_list=list(), ann_folder: str="ann") -> None:
        """
        Saves view to ann files.
        Args:
            column_list: list of columns to save, default all
            ann_folder: ann folder name
        """
        folder_name = os.path.join(os.getcwd(), ann_folder)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        if len(column_list) == 0:
            column_list = self.columns
        for sentence in self.sentences:
            cnt = 0
            data = ""
            words = sentence["words"]
            file_name = str(sentence["id"]) + ".ann"
            for word in words:
                cnt += 1
                data += "T" + str(cnt) + "\t" + "\t".join(
                    [word[column] for column in column_list]) + "\n"
            if len(data) > 0:
                with open(os.path.join(folder_name, file_name), "w") as f:
                    f.write(data)
