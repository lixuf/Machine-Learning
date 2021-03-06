total_data={} #全部数据存储 text aspect
mode=0 #划分数据集与训练集
maxlen=80#最长长度
order=['from','to','polarity','term']#aspect格式 

#读取xml并存储 索引保持str
import  xml.dom.minidom
dom = xml.dom.minidom.parse('Laptop_Train_v2.xml')
root = dom.documentElement
all_data = root.getElementsByTagName('sentence')
counter=0
for i in all_data:
    id=i.getAttribute("id")
    counter+=1
    data={}
    text=i.getElementsByTagName('text')
    for i1 in text:
        data['text']=i1.firstChild.data
    aspectterm=i.getElementsByTagName('aspectTerms')
    as_data=[]
    for i2 in aspectterm:
        aspect=i2.getElementsByTagName('aspectTerm')     
        for i1 in aspect:
            ass_data=[]
            ass_data.append(i1.getAttribute("from"))
            ass_data.append(i1.getAttribute("to"))
            ass_data.append(i1.getAttribute("polarity"))
            ass_data.append(i1.getAttribute("term"))
            as_data.append(ass_data)
    data['aspect']=as_data
    total_data[id]=data
print('counter',counter)
print('len(total_data)',len(total_data))
print(total_data["76"])

#repair
#1874 1650 1225
total_data['1650']['text']='With the macbook pro it comes with free security software to protect it from viruses and other intrusive things from downloads and internet surfing or emails.'
total_data['1650']["aspect"][0][3]="security software"
total_data['1225']['text']='The computer was shipped to their repair depot on june 24 and returned on July 2 seems like a short turn around time except the computer was not repaired when it was returned.'
total_data['1874']['text']='I feel that it was poorly put together, because once in a while different plastic pieces would come off of it.'
total_data['1874']["aspect"][0][3]="plastic pieces"


#process

#from to 延续 [ ， ) |分词
import keras
for i in total_data.keys():
    total_data[i]['text']=keras.preprocessing.text.text_to_word_sequence(total_data[i]['text'])
    for i1 in range(len(total_data[i]["aspect"])):
        total_data[i]['aspect'][i1][3]=keras.preprocessing.text.text_to_word_sequence(total_data[i]['aspect'][i1][3])
        total_data[i]['aspect'][i1][0]=total_data[i]['text'].index(total_data[i]['aspect'][i1][3][0])
        total_data[i]['aspect'][i1][1]=total_data[i]['text'].index(total_data[i]['aspect'][i1][3][-1])+1   
print(total_data["76"])


#w2v
import os
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
import multiprocessing
def word2vec(in_txt):
    path = get_tmpfile("word2vec_lstm.model")
    model = Word2Vec(in_txt, size=150, window=6, min_count=0,
                     workers=multiprocessing.cpu_count())
    model.save("word2vec_lstm.model")
if not (os.path.exists("word2vec_lstm.model")):
    print("start train word2vec......")
    text=[]
    for i in total_data.keys():
        text.append(total_data[i]["text"])
    print('len(text)',len(text))
    word2vec(text)
model=Word2Vec.load("word2vec_lstm.model")

#建立词表 id是int
import copy
id2word = {i+1:j for i,j in enumerate(model.wv.index2word)}
word2id = {j:i for i,j in id2word.items()}
total_data_un=copy.deepcopy(total_data)
for i in total_data_un.keys():
    for i2 in range(len(total_data_un[i]["text"])):
        total_data_un[i]["text"][i2]=word2id[total_data_un[i]["text"][i2]]
print(total_data_un["2334"]["text"][2])

#转化为词向量
for i in total_data.keys():
    senten=[]
    if maxlen<len(total_data[i]["text"]):
        maxlen=len(total_data[i]["text"])
    for i2 in range(0,len(total_data[i]["text"])):
        vector = model.wv[total_data[i]["text"][i2]]
        senten.append(vector)
    total_data[i]["text"]=senten
print('maxlen',maxlen)
print(total_data_un["2334"]["text"][2])

#padding
import numpy as np
print("padding....")
def seq_padding(x, padding=0):
    return np.array([
        np.concatenate([x, [padding] * (maxlen - len(x))]) if len(x) < maxlen else x 
    ])
for i in total_data.keys():
    total_data[i]["text"]=seq_padding( total_data[i]["text"],np.zeros(shape=(150,)))
for i in total_data_un.keys():
    total_data_un[i]["text"]=seq_padding( total_data_un[i]["text"])
print(total_data["76"])
print(total_data_un["76"])

#划分数据集
import numpy as np
import json
if not os.path.exists('../random_order_vote.json'):
    random_order =list(total_data.keys())
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('../random_order_vote.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('../random_order_vote.json'))
train_data = [total_data[j] for i, j in enumerate(random_order) if i % 5!= mode]
test_data = [total_data[j] for i, j in enumerate(random_order) if i % 5 == mode]
train_data_un = [total_data_un[j] for i, j in enumerate(random_order) if i % 5!= mode]
test_data_un = [total_data_un[j] for i, j in enumerate(random_order) if i % 5 == mode]
print('len(test_data)',len(test_data))
print('len(train_data)',len(train_data))
print('len(test_data_un)',len(test_data_un))
print('len(train_data_un)',len(train_data_un))
print(len(test_data_un[6]["text"]),len(test_data[6]["text"]))
print(test_data_un[6])


