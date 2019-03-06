# -*- coding: UTF-8 -*-
#(C) Copyright 2016 Meanotek AI

"""
Выделение сущностей из SentenceView, создание нового нода entity
в графе.
Главная функция entity_parser(sentv, tip) -> None
"""
from neuthink.graph import basics as graphb

def block_entity(times, graph, sent):
    """
    По выделенному фрагменту востанавливаем текст и создаем нод
    описания сущности
    """
 
    #~ print (times)
    li = []
    i = 0
    if len(times) > 0:
        if "pos_start" in times[0]:
            while i < len(times):
                dd = times[i]
                li.append(dd["word"])
                if i < len(times) - 1:
                    dell = int(times[i + 1]["pos_start"]) - int(dd["pos_end"])
                    li.append(" " * dell)
                i += 1
        else:
            while i < len(times):
                dd = times[i]
                li.append(dd["word"])
                li.append(" ")
                i += 1

    if len(li) > 0:
        if li[-1] == " ":
            li = li[:-1]
        cur_entity = graphb.Node(graph, {})
        cur_entity["type"] = "entity"
        cur_entity["text"] = "".join(li)
        if "pos_start" in times[0] and "pos_end" in times[0]:
            cur_entity["pos_start"] = times[0]["pos_start"]
            cur_entity["pos_end"] = times[-1]["pos_end"]
        cur_entity["entity_subtype"] = times[0]["tag"]
        if "date" in times[0]["tag"]:
            cur_entity["entity_type"] = "date"
        if "time" in times[0]["tag"]:
            cur_entity["entity_type"] = "time"
        sent.Connect(cur_entity)


def entity_parser(sentv, tip, word_name) -> None:
        """
        Creates new entity based on words with same class
        within sentence.
        Выделяем из графа сходные типы и группируем их по спискам.
        sentv - Экземпляр класса SentenceView
        tip - Просматриваемый тип
        """
        ents = []
        times = []
        sentv.columns = ["type", "pos_start", "pos_end", "word"]
        #~ print (len(sentv.sentences))
        print(sentv)
        for sent in sentv:
            #print(len(sent.Children({'type':word_name})))
            for word in sent.Children({'type':word_name}):
                #~ print(word)
                if len(times) == 0 and word[tip] != "OTHER" and word[tip] != "other":
                   # print ("yes-1")
                    dword = {}
                    dword["word"] = word[word_name]
                    if "pos_start" in word and "pos_end" in word:
                        dword["pos_start"] = word["pos_start"]
                        dword["pos_end"] = word["pos_end"]
                    dword["tag"] = word[tip]
                    times.append(dword)

                elif  len(times) > 0 and word[tip] != "OTHER" and word[tip] != "other" and word[tip] == times[0]["tag"] :
                   # print ("yes-2")
                    dword = {}
                    dword["word"] = word[word_name]
                    if "pos_start" in word and "pos_end" in word:
                        dword["pos_start"] = word["pos_start"]
                        dword["pos_end"] = word["pos_end"]
                    dword["tag"] = word[tip]
                    times.append(dword)

                elif word[tip] == "OTHER" or word[tip] == "other":
                    if len(times) > 0:
                    #    print ("yes1")
                        #~ ents.append(times)
                        block_entity(times, sentv.parent_graph, sent)
                        times = []
                elif  len(times) > 0 and word[tip] != times[0]["tag"]:
                    #print ("yes11111")
                    if len(times) > 0:
                        block_entity(times, sentv.parent_graph, sent)
                        times = []
                    dword = {}
                    dword["word"] = word[word_name]
                    if "pos_start" in word and "pos_end" in word:
                        dword["pos_start"] = word["pos_start"]
                        dword["pos_end"] = word["pos_end"]
                    dword["tag"] = word[tip]
                    times.append(dword)



                else:
                    pass
            else:
                if len(times) > 0:
                        #~ print ("yes2")
                        #~ ents.append(times)
                        block_entity(times, sentv.parent_graph, sent)
                        times = []
        if len(times) > 0:
            #~ print (yes3)
            #~ print (times)
            #~ ents.append(times)
            block_entity(times, sentv.parent_graph, sentv.sentences[-1])

