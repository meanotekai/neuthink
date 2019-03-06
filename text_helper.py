#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import re
import codecs

separators = [u";", u",", u".", u"(", u")", u"!", u"?", u":", u"/",
    u"\r", u"\n", u"\t", u"*", u"~", u"+", u"=", u'"', u" ", u" ",
    u"&", u" ", u"#", u"/", u" ", u" ", u" ", u" ", u"“", u"”",u"-",u"."  ]

res_sentence = re.compile('''(?<!\w\.\w.)(?<![А-ЯA-Z][а-яa-z]\.)(?<=\.|\?)\s''')
res_empty = re.compile('''[ ]{2,}''')

res_fio1 = re.compile('''[A-ZА-Я]{1}\.?[ ]{0,8}([A-ZА-Я]{1}\.?[ ]{0,8})?[A-ZА-Я]{1}[a-zа-я]{2,20}''')
res_fio2 = re.compile('''[A-ZА-Я]{1}[a-zа-я]{2,20}[ ]{0,8}[A-ZА-Я]{1}\.?[ ]{0,8}([A-ZА-Я]{1}\.?[ ]{0,8})?''')
res_2_3 = re.compile('''([а-яa-z]{1,3}\.)([а-яa-z]{1,3}\.)([а-яa-z]{1,3}\.)?''')

def load_dictionary_stop_words(langs):
    '''
    load stop words

    INPUTS
    lang: is tuple of part of filename, for example [RU,ENE]

    RETURNS

    stopw: is tuple all stop words from files
    '''
    stopw=[]
    for lang in langs:
        way = "/home/katriniz/Dropbox/paiment/word_with_languge/dics/%s_official_parts.dic" % lang
        fi = codecs.open(way, "r", "utf-8")
        stw = map(lambda i: i.strip(), fi.readlines())
        fi.close()
        stopw.extend(stw)
    return tuple(stopw)

def word_dic(wrd, par):
    res = []
    if wrd not in [" ", "   "]:
        word = {}
        word["word"] = wrd
        word["pos_end"] = par[1]
        word["pos_start"] = par[0]
        res = [word]
    return res

def clear_text(st):
    li = ['\n', '\t', '\r', '\a', '\0', '\a', '\b',  '\v', '\f' ]
    for obj in li:
        st = st.replace(obj, " ")
    st = " ".join(st.split())
    st = res_empty.sub("", st)
    return st

def load_words_save_separators(st):
    """
    Из строки получить список слов с их индексами
    """
    #~ print(st)
    #~ st = clear_text(st)
    #~ print (st)
    i = 0
    stat = 0
    word = []
    li = []
    for simv in st:
        if simv in separators:
            if len(word) > 0:
                wrd = "".join(word)
                inds = (stat, i)
                word = []
                li.extend(word_dic(wrd, inds))
                inds = (i, i+1)
                li.extend(word_dic(simv, inds))
            else:
                inds = (i, i+1)
                li.extend(word_dic(simv, inds))
            stat = i+1
        else:
            word.append(simv)
        i = i + 1
    if len(word) > 0:
        inds = (stat, len(st))
        wrd = "".join(word)
        li.extend(word_dic(wrd, inds))
    return li



# РАЗНЕСЕНИЕ ПО ПРЕДЛОЖЕНИЯМ
def pre_conv(st):
    st  = st.replace(u"пер.",u"пер% ").replace(u"ул.",u"ул% ").replace(u"п.",u"п%").replace(u"г.",u"г%").replace(u"пос.",u"пос%").replace(u"„",u"«").replace(u"“",u"»").replace(u"«",u"« ").replace(u"&#1201;",u"") .replace(u"кв.м.",u"кв%м%") .replace(u"\n", u" Aaan%").replace(u"\r", u" Aaar%")
    return st

def recover(st):
    st = st.replace(u"пер%" , u"пер. ").replace(u"ул%" , u"ул. ").replace(u" п%",u"п. ").replace(u"г%",u"г. ").replace(u"пос%",u"пос.") .replace(u"кв%м%", u"кв.м.") .replace(u" Aaan%", u"\n").replace(u" Aaar%", u"\r")
    return st

patt = {"кв.м.":"####", "пер.":"####" , "ул.":"###", "пос.":"####", "кв.км.":"######", "\n":"##", "\r":"##", "\t":"##"}


def text_into_sentences2(st,sent_separator=[".", "!", "?"]):
    st_sens = st
    patterns = [res_fio1, res_fio2, res_2_3]
    li = []
    for elem in patterns:
        r23 = elem.finditer(st)
        for r in r23:
            p1, p2 = r.span()
            fragm = st[p1:p2]
            nst = st[:p1] + "#"*len(fragm) + st[p2:]
            #~ li.append((fragm, (p1,p2)))
            st = nst
        #~ print (st)
    #~ print("_______________________")
    for pat in patt:
        rr = True
        #~ print(pat)
        while rr:
            r = st.find(pat)
            #~ print (pat, r)
            if r > -1:
                how = len(pat)
                nst = st[:r] + patt[pat] + st[r+how:]
                st = nst
                #~ li.append((pat, (r, r + how)))
            else:
                rr = False
    #~ print(st)
    #~ print (li)
    #~ li.sort(key=lambda x: x[1][0], reverse=False)
    #~ print(li)
    points = []
    st_sens = st
    pp = sent_separator
    i = 0
    slen = 0
    for simv in st:
        slen=slen+1
        if simv in pp and slen>3:
            points.append(i)
            slen = 0
        i+= 1
    #~ print (points)
    sents = []
    sta = 0
    for poin in points:
          sen = st_sens[sta+1:poin]
          #print(poin)
          #print (sen +"|"+st_sens[poin])
          sents.append((sta+1,poin))
          sta = poin
          #~ print(sen)
    else:
        #sen = st_sens[sta:]
        sents.append((sta,-1))
        #print(sents[-1])
    return sents


def text_into_sentences(st):
    txt2 = pre_conv(st)
    #~ sentences = res_sentence.split(txt2)
    res_fi0 = res_fio1.finditer(txt2)
    fragments = []
    for pp in res_fi0:
         pos1, pos2 = pp.span()
         find_fragm =  txt2[pos1:pos2]
         #~ print (find_fragm)
         repl_fragm = find_fragm.replace(".", "%^%")
         #~ print (repl_fragm)
         fragments.append((find_fragm, repl_fragm))
         txt2 = txt2.replace( find_fragm, repl_fragm )
         #~ print (pos1, pos2)
    #~ print (txt2)
    sentences = res_sentence.finditer(txt2)
    #~ print("___^____________^_____")
    sta = 0
    li = []
    #~ print ("FINND PARES")
    for sent in sentences:
        pos1, pos2 = sent.span()
        #~ print (sta, pos1)
        #~ print (txt2[sta:pos1])
        li.append((sta, pos1))
        #~ print(st[sta:pos1])
        sta = pos2
    if len(st) > 0 and len(li) == 0:
        li.append((0, len(st)-1))
    #~ print ("FINND PARES")
    return li

def get_sentences(words, text, sent_separator=[".", "!", "?"]):
    sents = text_into_sentences2(text,sent_separator=sent_separator)
    #~ print (sents)
    ss = [[] for x in sents]
    for word in words:
        i = 0
        fin = word["pos_end"]
        sta = word["pos_start"]
        for sent in sents:
            if sta >= sent[0] and fin <= sent[1] :
                ss[i].append(word)
            i = i + 1
    res = [li for li in ss if len(li) > 0]
    return res

def open_text(fname):
    f = open(fname, "r")
    a = f.read()
    f.close()
    return a

if  __name__ ==  "__main__" :
    dd = '''ТО «Вираж» была организована в 2009 году, в городе Талдыкорган по адресу: ул. Ж. Жабаева. Общая площадь, занимаемая территорией СТО, составляет 26000 кв.м. Предприятие оказывает услуги в сфере технического обслуживания и ремонта легковых автомобилей как отечественного, так и импортного производства.
    Рост количества автомобилей в городе Талдыкорган приводит к необходимости развития транспортной системы страны. Также это влечет за собой повышенный спрос на услуги квалифицированных автосервисов. Поэтому появление все большего числа станций технического обслуживания вполне закономерно.'''
    #~ sents= text_into_sentences2(dd)
    #~ for st, fin in sents:

        #~ print ("sentens = ", dd[st:fin])

    #~ words = load_words_save_separators(dd)
    #~ print (words)
    #~ hh = get_sentences(words, dd)
    #~ print (hh)

