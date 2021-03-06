import sys
import numpy as np
import gensim
import jieba
from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
import multiprocessing
import warnings
TaggededDocument = gensim.models.doc2vec.TaggedDocument
"""
with open('stop_words.txt','r',encoding='GBK') as stopdata:
    stoplist=stopdata.readlines()




def getdata():
    with open('语料2.txt','r',encoding='utf-8') as indata:
        train_data=indata.readlines()
    return train_data
def cleandata(corpus):
        cacha_data=open('语料3.txt','w',encoding='utf-8')
        punctuation = .,?!:;(){}[]
        corpus = [z.lstrip('1,') for z in corpus]
        corpus = [z.lstrip('0,') for z in corpus]
        corpus = [z.strip('"') for z in corpus]
        corpus = [z.replace('\n','') for z in corpus]
        corpus = [z.replace('<br />', ' ') for z in corpus]
        for z in corpus:
           cacha_data.write(z)
           cacha_data.write('\n')
        cacha_data.close()
      return corpus
def add_tab(indata):
    cacha_data=open('语料分词.txt','w',encoding='utf-8')
    for n2 in indata:
        curLine = ' '.join(list(jieba.cut(n2)))
        cacha_data.write(curLine)
    cacha_data.close()
    with open ('语料分词.txt','r',encoding='utf-8') as datain:
        docs=datain.readlines()
        for idx in list(range(0, len(docs))):
            docs[idx] = ' '.join([word for word in docs[idx].split() if word not in stoplist])
        docs = [doc for doc in docs if len(doc) > 0]

"""
def mid():
    with open('语料分后.txt','r',encoding='utf-8') as indata:
        docs=indata.readlines()
    data = []
    for i, text in enumerate(docs):  
        word_list = text.split(' ')  
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()  
        document = TaggededDocument(word_list, [str(i)])
        data.append(document)  
    return data
def train_data(indata):
    print('s')
 
    path = get_tmpfile("doc2vec_dm_100_1.model")
    
    model_dm = gensim.models.Doc2Vec(indata, min_count=5, window=8, size=100,  workers=multiprocessing.cpu_count())
    
   
    model_dm.save("doc2vec_dm_100_1.model")
def runmodel():
    warnings.filterwarnings(action='ignore', category=UserWarning,module='gensim')
    model = Doc2Vec.load("doc2vec_dm_100_1.model")
    print('与标签0相似的句子')
    print(model.docvecs.most_similar(0))
    print('标签0与1相似的程度')
    print(model.docvecs.similarity(0,1))
    
  

#indata=getdata()
#indata=cleandata(indata)

#able_train_data=add_tab(indata)
train_data(mid())

runmodel()
