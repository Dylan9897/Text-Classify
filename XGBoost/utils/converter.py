"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/4/19 11:22
@Email : handong_xu@163.com
"""
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def dump(target,object):
    with open(target,'wb') as fw:
        pickle.dump(object,fw)

def undump(object):
    with open(object,'rb') as fw:
        obj = pickle.load(fw)
    return obj


class Converter():
    def __init__(self):
        pass

    def operate(self,seq):
        while '  ' in seq:
            seq = seq.replace('  ',' ')
        return seq


    def tfvectorize(self,words,test=None):
        v = TfidfVectorizer(binary=False,decode_error='ignore',stop_words='english')
        if test == None:
            print('train tf-idf')
            print("sadad")
            data = v.fit_transform(words.values.astype('U'))
            print("#@###")
            dump('ckpt/tfidf2.model',v)
        else:
            print('load tf -idf')
            v = undump('ckpt/tfidf2.model')
            data = v.transform(words.values.astype('U'))
        return data

