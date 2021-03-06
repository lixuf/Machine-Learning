from keras.layers.core import Activation,Dense,Dropout,SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils import multi_gpu_model
from keras.preprocessing import sequence
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

def step1_combine():
    intag=[]
    inwords=[]
    with open("语料.txt","r",encoding="GB18030") as file:
        lines=file.readlines()
        for line in lines:
            if line[0]=="1":
                intag.append(1)
                line=line.lstrip("1,")
                line=line.strip()
                inwords.append(line)
            elif line[0]=="0":
                intag.append(0)
                line=line.lstrip("0,")
                line=line.strip()
                inwords.append(line)
    print(len(intag))
    print(len(inwords))
    f=xlwt.Workbook()
    sheet1=f.add_sheet("合并",cell_overwrite_ok=True)
    r=xlrd.open_workbook(filename="online_shopping_10_cats.xls")
    sheet2=r.sheet_by_index(0)
    intag1=sheet2.col_values(1)
    inwords1=sheet2.col_values(2)
    inclass=sheet2.col_values(0)
    print(len(intag1))
    print(len(inwords1))
    print(len(inclass))
    count=0
    for i in range(0,len(intag)):
        sheet1.write(i,1,intag[i])
        sheet1.write(i,2,inwords[i])
        count=count+1
    for i in range(count,len(intag1)):
        sheet1.write(i,1,intag1[i])
        sheet1.write(i,2,inwords1[i].strip())
        sheet1.write(i,0,inclass[i])
    f.save("combine.xls")

def step2_pre_processing(suffix_stop=""):
    r=xlrd.open_workbook(filename="combine.xls")
    sheet1=r.sheet_by_index(0)
    intag=sheet1.col_values(1)
    inclass=sheet1.col_values(0)
    inwords=sheet1.col_values(2)
    print(len(intag))
    print(len(inwords))
    stop_word=[]
    with open("stop_word"+suffix_stop+".txt","r",encoding="GB18030") as file:
        instop_words=file.readlines()
        for i in instop_words:
           stop_word.append(i.strip())
    w=xlwt.Workbook()
    sheet2=w.add_sheet("准备文件",cell_overwrite_ok=True)
    count=0
    word_freqs=collections.Counter()
    vocall=[]
    inword=[]
    for word in inwords:
        word=jieba.cut(word)
        for i in word:
            word_freqs[i]+=1
    for word in inwords:
        word=jieba.cut(word)
        new_word=""
        for i in word:
            if word_freqs[i]>4:
                if i not in vocall:
                    vocall.append(i)
                if i not in stop_word:
                    new_word=new_word+i+' '
        new_word=new_word.strip()
        inword.append(new_word)
        sheet2.write(count,2,new_word)
        count=count+1
    print(count)
    print(len(intag))
    maxlen=0
    sumlen=0
    esumlen=0
    num=0
    num=len(inword)
    for i in inword:
        sumlen=sumlen+len(i)
        if len(i)>maxlen:
            maxlen=len(i)
    esumlen=sumlen/num
    outclass=[]
    outclass_count=collections.Counter()
    intag_count=collections.Counter(intag)
    n=6
    sheet2.write(0,11+n,"总个数")
    sheet2.write(0,10+n,"平均长度")
    sheet2.write(0,9+n,"最长长度")
    sheet2.write(1,11+n,num)
    sheet2.write(1,10+n,esumlen)
    sheet2.write(1,9+n,maxlen)
    sheet2.write(0,8+n,"分割线")
    sheet2.write(1,8+n,"20804")
    sheet2.write(0,6+n,"情感")
    sheet2.write(1,6+n,"1")
    sheet2.write(2,6+n,"0")
    sheet2.write(0,7+n,"数量")
    sheet2.write(0,6,"词总数")
    sheet2.write(1,6,len(vocall))
    for i in range(0,len(vocall)):
        sheet2.write(i,4,vocall[i])
        sheet2.write(i,5,word_freqs[vocall[i]])
    if intag_count["1"]==0:
        sheet2.write(1,7+n,intag_count[1])
        sheet2.write(2,7+n,intag_count[0])
    else:
        sheet2.write(1,7+n,intag_count["1"])
        sheet2.write(2,7+n,intag_count["0"])
    for i in range(0,len(intag)):
        sheet2.write(i,1,intag[i])
    for i in range(0,len(inclass)):
        sheet2.write(i,0,inclass[i])
        outclass_count[inclass[i]]+=1
        if inclass[i] not in outclass:
            outclass.append(inclass[i])
    for i in range(0,len(stop_word)):
        sheet2.write(i,3,stop_word[i])
    sheet2.write(0,4+n,"种类")
    sheet2.write(0,5+n,"数量")
    for i in range(0,len(outclass)):
        sheet2.write(i+1,4+n,outclass[i])
        sheet2.write(i+1,5+n,outclass_count[outclass[i]])
    w.save("pre处理"+suffix_stop+".xls")
def spilt_words(suffix=""):
    r=xlrd.open_workbook(filename="pre处理"+suffix+".xls")
    sheet1=r.sheet_by_index(0)
    intag=sheet1.col_values(1)
    inwords=sheet1.col_values(2)
    inclass=sheet1.col_values(0)
    elen=sheet1.col_values(16)
    elen=elen[1]
    elen=int(elen)
    print(elen)
    out1_words=[]
    out2_words=[]
    out1_tag=[]
    out2_tag=[]
    out1_class=[]
    out2_class=[]
    maxlen=elen
    for i,precess_words in enumerate(inwords):
        precess_words=inwords[i]
        swich=0
        while len(precess_words)>maxlen:
            if swich==0:
                mid=precess_words[0:maxlen]
                out1_words.append(mid)
                out1_tag.append(intag[i])
                out1_class.append(inclass[i])
                out2_words.append(mid)
                out2_tag.append(intag[i])
                out2_class.append(inclass[i])
                swich=1
            else:
                mid=precess_words[0:maxlen]
                out2_words.append(mid)
                out2_tag.append(intag[i])
                out2_class.append(inclass[i])
            precess_words=precess_words[maxlen:len(precess_words)].strip()
            
        if len(precess_words)!=0:
            if swich==0:
                out1_words.append(precess_words)
                out2_words.append(precess_words)
                out1_tag.append(intag[i])
                out2_tag.append(intag[i])
                out1_class.append(inclass[i])
                out2_class.append(inclass[i])
            elif len(precess_words)<maxlen/2:
                out2_words.append(precess_words)
                out2_tag.append(intag[i])
                out2_class.append(inclass[i])
    print("start save.........")
    w=xlwt.Workbook()
    sheet2=w.add_sheet("准备文件",cell_overwrite_ok=True)
    sheet2.write(0,10,maxlen)
    sheet2.write(0,11,"切割丢掉")
    sheet2.write(1,11,len(out1_tag))
    sheet2.write(0,12,"切割扩充")
    sheet2.write(1,12,len(out2_tag))

    swich2=0
    for i in range(0,len(out2_words)):
        if i==65000:
            swich=1
            break
        sheet2.write(i,3,out2_class[i])
        sheet2.write(i,4,out2_tag[i])
        sheet2.write(i,5,out2_words[i])
    if swich==1:
        sheet2.write(0,13,"1")
        
        for i in range(65000,len(out2_words)):
            sheet2.write(i-65000,6,out2_class[i])
            sheet2.write(i-65000,7,out2_tag[i])
            sheet2.write(i-65000,8,out2_words[i])
    else:
        sheet2.write(0,13,"0")
        
    for i in range(0,len(out1_words)):
        sheet2.write(i,0,out1_class[i])
        sheet2.write(i,1,out1_tag[i])
        sheet2.write(i,2,out1_words[i])
    w.save("切割"+suffix+".xls")
def model(suffix=""):
    xtest=[]
    ytest=[]
    for round in range(0,2):
        r=xlrd.open_workbook(filename="切割"+suffix+".xls")
        r_vocall=xlrd.open_workbook(filename="pre处理"+suffix+".xls")
        sheet1=r.sheet_by_index(0)
        sheet1_vocall=r_vocall.sheet_by_index(0)
        invocall=sheet1_vocall.col_values(4)
        vocall_size=len(invocall)
        print(len(invocall))
        inwords=sheet1.col_values(2+round*3)
        inclass=sheet1.col_values(0+round*3)
        intags=sheet1.col_values(1+round*3)
        print(len(inwords))
        print(len(intags))
        print(len(inclass))
        for n1 in range(0,len(inwords)):
            if len(inwords[n1])==0:
                inwords=inwords[:n1]
                intags=intags[:n1]
                inclass=inclass[:n1]
                break
        print(len(inwords))
        print(len(intags))
        print(len(inclass))
        if round==1:
            other=sheet1.cell(0,13).value
            print(type(other))
            other=int(other)
            if other==1:
                inwords=inwords+sheet1.col_values(8)
                intags=intags+sheet1.col_values(7)
                inclass=inclass+sheet1.col_values(6)
        for n1 in range(0,len(inwords)):
            if len(inwords[n1])==0:
                inwords=inwords[:n1]
                intags=intags[:n1]
                inclass=inclass[:n1]
                break

        maxlen=sheet1.cell(0,10).value
        tokenizer=Tokenizer(num_words=vocall_size)
        tokenizer.fit_on_texts(invocall)
        words=tokenizer.texts_to_sequences(inwords)
        maxlen=0
        for n1 in words:
            if len(n1)>maxlen:
                maxlen=len(n1)
        word=pad_sequences(words,padding='post',maxlen=maxlen)
        if round==0:
            xtrain,xtest,ytrain,ytest=train_test_split(word,intags,test_size=0.2,random_state=42)
        else:
            xtrain,xtest1,ytrain,ytest1=train_test_split(word,intags,test_size=0.4,random_state=42)
        embedding_size=150
        hidden_layer_size=64
        batch_size=128
        num_epochs=3
        print(ytrain[2])
        print(type(ytrain[2]))
        model=Sequential()
        model.add(Embedding(vocall_size,embedding_size,input_length=maxlen))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(hidden_layer_size,dropout=0.2,recurrent_dropout=0.2))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        model.summary()
        
        history=model.fit(xtrain,ytrain,epochs=10,batch_size=64)
        loss,accuracy=model.evaluate(xtest,ytest)
        print(loss,accuracy)
        """
        plt.subplot(211)
        plt.title("Accuracy"+suffix)
        plt.plot(history.history['acc'],color="g",label="Train")
      
        plt.legend(loc="best")

        plt.subplot(212)
        plt.title("Loss")
        plt.plot(history.history['loss'],color="g",label="Train")
       
        plt.legend(loc="best")

        plt.tight_layout()
        plt.show()
        """

        w=xlwt.Workbook()
        sheet2=w.add_sheet("准备文件",cell_overwrite_ok=True)
        sheet2.write(0,0,"predict")
        sheet2.write(0,1,"ytest")
        sheet2.write(0,2,"xtest")
        sheet2.write(0,3,"loss")
        sheet2.write(1,3,loss)
        sheet2.write(0,4,"acc")
        sheet2.write(1,4,accuracy)
        """
        for index,i in enumerate(xtest):
            ypred=model.predict(i)
            sent=tokenizer.sequences_to_texts(i)
            sheet2.write(1+i,0,ypred)
            sheet2.write(1+i,1,ytest[index])
            sheet2.write(1+i,2,sent)
        """
        if round==0:
            w.save("result切割"+suffix+".xls")
        else:
            w.save("result扩充"+suffix+".xls")


model("3")



    
           


