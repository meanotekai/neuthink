from typing import List

def load_file(name, encoding='utf-8'):
    '''just load text file'''
    return open(name,errors="ignore",encoding=encoding).read()

def save_file(name, string):
    '''just save text file'''
    return open(name, "w", errors="ignore",encoding='utf-8').write(string)

def load_lines(name:str) -> List[str]:
    '''just load lines from text file'''
    f = open(name)
    lines = f.readlines()
    f.close()
    return [x.strip() for x in lines]

def save_lines(name:str, lines:List[str]):
    '''just saves lines to text file'''
    f = open(name,"w",encoding='utf-8')
    for x in lines:
      f.write(x+'\n')
    f.close()


def usplit(text, separators=[], keep_separators=True, keep_space=False, keep_new_line=False,  return_positions=True):
    '''The ultimate split function. Alternative to split.
       Args:
         text: text to split
         separators: list of separators, default is all punctionation and space
         keep_separators: keep separators after split or not
         keep_space: keep space or remove
         return_positions: return character spans of each word or not
    '''

    def update_buffer(buffer,word_buffer,i, x):
        if len(word_buffer)>0:
            if keep_separators or (not word_buffer in separators):
             if return_positions:
                buffer.append((word_buffer,i-len(word_buffer),i))
             else:
                buffer.append(word_buffer)
        if keep_new_line:
            return  (''  if ((x==" ")) and (not keep_space) else x)
        else:
            return  (''  if ((x==" ") or x=="\n") and (not keep_space) else x)

    if separators==[]:
        separators = set([u";", u",", u".", u"(", u")", u"!", u"?", u":", u"/",
    u"\r", u"\n", u"\t", u"*", u"~", u"+", u"=", u'"', u" ", u" ",
    u"&", u" ", u"#", u"/", u" ", u" ", u" ", u" ", u"“", u"”",u"-",u".","«","»","…"])
    else:
        separators = set(separators)
    buffer = []
    word_buffer=''
    state = 0
    for i,x in enumerate(text):
        if x in separators:
            if state==1:
                word_buffer = update_buffer(buffer, word_buffer, i, x)
            if state==0:
                word_buffer = update_buffer(buffer, word_buffer, i, x)
                state=1
        else:
            if state==1:
                update_buffer(buffer, word_buffer, i, x)
                word_buffer =''
                state=0

            word_buffer = word_buffer + x
    if return_positions:
        buffer.append((word_buffer,len(text)-len(word_buffer),len(text)))
    else:
        buffer.append(word_buffer)

    return buffer
