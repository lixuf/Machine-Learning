import pandas as pd
import numpy as np
import jieba
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
#导入w2v
"""
w2vmodel = KeyedVectors.load_word2vec_format('word2vec_400.model')
"""

incsv=pd.read_csv('online_shopping_10_cats.csv',encoding='GB18030',names=['type','sort','sentence'])
train_data = np.array(incsv.sentence)
x_train1=train_data.tolist()
train_data=np.array(incsv.sort)
y_train=train_data.tolist()
x_test=list()
y_test=list()
with open('测试.txt','r',encoding='GB18030') as filep:
    lines=filep.readlines()
    for line in lines:
        x_test.append(line[2:].strip())
        if(line[0]=='1'):
            y_test.append(1)
        else:
            y_test.append(0)

x_train=list()

for line in x_train1:
    x_train.append(str(line).strip())




for n1 in range(len(x_train)):
    if len(x_train[n1])>200:
        x_train[n1]=x_train[n1][0:198]
for n1 in range(len(x_test)):
    if len(x_test[n1])>200:
        x_test[n1]=x_test[n1][0:198]
    



from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
#切词
list_cut=x_train
x_train=list()
for lines in list_cut:
    lines.strip()
    word_list=jieba.cut(lines)
    words=''
    for word in word_list:
        words=words+word+' '
    x_train.append(words)
list_cut=x_test
x_test=list()
for lines in list_cut:
    lines.strip()
    word_list=jieba.cut(lines)
    words=''
    for word in word_list:
        words=words+word+' '
    x_test.append(words)

        



#转换成索引
tokenizer=Tokenizer(num_words=5000)
tokenizer.fit_on_texts(x_train)

x_train=tokenizer.texts_to_sequences(x_train)
x_test=tokenizer.texts_to_sequences(x_test)

vocab_size=len(tokenizer.word_index)+1
#补齐
maxlen=200
x_train=pad_sequences(x_train,padding='post',maxlen=maxlen)
x_test=pad_sequences(x_test,padding='post',maxlen=maxlen)
print(x_train[1])

#embbading
embedding_dim=64

model=Sequential()
model.add(layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=maxlen))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

history=model.fit(x_train,y_train,epochs=2,batch_size=10)
loss,accuracy=model.evaluate(x_test,y_test)
print(loss,accuracy)