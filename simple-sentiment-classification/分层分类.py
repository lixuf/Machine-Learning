from keras.layers.core import Activation,Dense,Dropout,SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Masking
from keras.models import Sequential
from keras.utils import multi_gpu_model
from keras.preprocessing import sequence
import keras
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
import jieba
import xlwt
from keras.preprocessing.text import Tokenizer
import xlrd
from gensim.models import Word2Vec
import gensim
from attention_encoder import AttentionDecoder as AT

def CNN(xtrain,ztrain,xtest,ztest,classes,vocall,embedding_matrix,ytrain,ytest):
    print(classes)
    xtrain_copy=[]
    xtest_copy=[]
    ztrain_copy=[]
    ztest_copy=[]
    for i in range(0,len(xtrain)):
        if ztrain[i]!="":
            xtrain_copy.append(xtrain[i])
            if ztrain[i]!=classes:
                ztrain_copy.append(0)
            else:
                ztrain_copy.append(1)
    for i in range(0,len(xtest)):
        if ztest[i]!="":
            xtest_copy.append(xtest[i])
            if ztest[i]!=classes:
                ztest_copy.append(0)
            else:
                ztest_copy.append(1)
    xtrain=np.array(xtrain_copy)
    ztrain=ztrain_copy
    xtest=np.array(xtest_copy)
    ztest=ztest_copy
    print(ztest[4])
    print(xtrain[0])
    print(type(xtrain))
    hidden_layer_size=64
    """
    model=Sequential()
    model.add(Embedding(input_dim=len(vocall)+1,output_dim=150,weights=[embedding_matrix],mask_zero=True))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(hidden_layer_size,dropout=0.2,recurrent_dropout=0.2))
    model.add(Dense(10))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
      
    history=model.fit(xtrain,ztrain,epochs=1,batch_size=64)
    """
    model=keras.models.load_model("分类"+str(classes)+".h5")
   
    loss,accuracy=model.evaluate(xtest,ztest)
    f1=xlwt.Workbook()
    sheetf1=f1.add_sheet("合并",cell_overwrite_ok=True)
    ypred=model.predict_classes(xtest[i],1)
    for i in range(0,len(ztest)):
        sheetf1.write(i,0,ztest[i])
        sheetf1.write(i,1,int(ypred[i][0]))
    f1.save("test.xls")
    print(loss,accuracy)
    model.save("分类"+str(classes)+".h5")

   

def model():
    r1=xlrd.open_workbook(filename="pre处理3train.xls")
    r2=xlrd.open_workbook(filename="pre处理3test.xls")
    r3=xlrd.open_workbook(filename="class.xls")
    sheet1=r1.sheet_by_index(0)
    sheet2=r2.sheet_by_index(0)
    sheet3=r3.sheet_by_index(0)
    classes=sheet3.col_values(0)
    xtrain=sheet1.col_values(2)
    ytrain=sheet1.col_values(1)
    ztrain=sheet1.col_values(0)
    xtest=sheet2.col_values(2)
    ytest=sheet2.col_values(1)
    ztest=sheet2.col_values(0)
    vocall_x=sheet1.col_values(4)
    vocall_y=sheet2.col_values(4)
    vocall=[]
    
    maxlen_test=0
    for i in range(0,len(vocall_x)):
        if vocall_x[i] not in vocall:
            vocall.append(vocall_x[i])
    for i in range(0,len(vocall_y)):
        if vocall_y[i] not in vocall:
            vocall.append(vocall_y[i])
    print(len(vocall),"vocall")
    allclass={}
    
    for i in range(0,len(classes)):
        if classes[i]=="":
            calsses=classes[:i]
            
            break
    print(len(classes),"classes")
    for i in range(0,len(classes)):
        allclass[classes[i]]=int(i)
        print(i,classes[i])
    for i in range(0,len(xtrain)):
        if xtrain[i]=="":
            if ytrain[i]=="":
                xtrain=xtrain[:i]
                ytrain=ytrain[:i]
                ztrain=ztrain[:i]
            break
        ytrain[i]=int(ytrain[i])
        ztrain[i]=allclass[ztrain[i]]
    for i in range(0,len(xtest)):
        if xtest[i]=="":
            xtest=xtest[:i]
            ytest=ytest[:i]
            ztest=ztest[:i]
            break
        ytest[i]=int(ytest[i])
        ztest[i]=allclass[ztest[i]]
    print("xtrain",len(xtrain),xtrain[0])
    print("ytrain",len(ytrain),ytrain[0])
    print("ztrain",len(ztrain),ztrain[0])
    print("xtest",len(xtest),xtest[0])
    print("ytest",len(ytest),ytest[0])
    print("ztest",len(ztest),ztest[0])





    maxlen=220
    vocall_new=[]
    modelw2v = gensim.models.KeyedVectors.load("word2vec_150_lstm.model")
    for i,word in enumerate(vocall):
        try:
           embedding_vector = modelw2v[word]
           vocall_new.append(word)
        except KeyError:
           pass
    print(len(vocall_new))
    vocall=vocall_new
    vocall_size=len(vocall)
    tokenizer=Tokenizer(num_words=vocall_size)
    tokenizer.fit_on_texts(vocall)
    xtrain=tokenizer.texts_to_sequences(xtrain)
    xtest=tokenizer.texts_to_sequences(xtest)
    xtrain=pad_sequences(xtrain,padding='post',maxlen=maxlen)
    xtest=pad_sequences(xtest,padding='post',maxlen=maxlen)

    embedding_matrix = np.zeros(shape=(vocall_size+1 ,150))
    for i,word in enumerate(vocall):
        try:
            embedding_vector = modelw2v[word]
            embedding_matrix[i+1,:] = embedding_vector
        except KeyError:
            pass

    print(xtrain[1])
    hidden_layer_size=64
    print(type(xtrain))
    
    for i in range(0,len(xtrain)):
        if len(xtrain[i])>maxlen:
            xtrain[i]=xtrain[:maxlen]

    for i in range(0,len(allclass)):
        CNN(xtrain,ztrain,xtest,ztest,i,vocall,embedding_matrix,ytrain,ytest)



    





    
    model=Sequential()
 
    model.add(Embedding(input_dim=len(vocall)+1,output_dim=150,weights=[embedding_matrix],mask_zero=True))
   
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(hidden_layer_size,dropout=0.2,recurrent_dropout=0.2))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
        
    history=model.fit(xtrain,ytrain,epochs=1,batch_size=64)
    loss,accuracy=model.evaluate(xtest,ytest)
    print(loss,accuracy)

    


model()