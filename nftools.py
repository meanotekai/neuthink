def split(string ,delimeter):
    '''this is helper function to avoid crazy pythonish behavior of split'''

    s = string.split(delimeter)
    s = list(filter(lambda x: len(x)>0, s))
    return s