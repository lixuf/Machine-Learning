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
from gensim.models import Word2Vec
import gensim
from attention_encoder import AttentionDecoder as AT
def step1_combine():
    f1=xlwt.Workbook()
    f2=xlwt.Workbook()
    sheetf1=f1.add_sheet("合并",cell_overwrite_ok=True)
    sheetf2=f2.add_sheet("合并",cell_overwrite_ok=True)
    r=xlrd.open_workbook(filename="combine.xls")
    sheet2=r.sheet_by_index(0)
    intag=sheet2.col_values(1)
    inwords=sheet2.col_values(2)
    inclass=sheet2.col_values(0)
    print(len(intag))
    print(len(inwords))
    print(len(inclass))
    count1=0
    ramdom1=8
    ramdom2=4
    count2=0
    
    for i in range(0,len(intag)):
        if i==ramdom1:
            ramdom1+=10
            sheetf2.write(count2,0,inclass[i])
            sheetf2.write(count2,1,intag[i])
            sheetf2.write(count2,2,inwords[i])
            count2+=1
        elif i==ramdom2:
            ramdom2+=10
            sheetf2.write(count2,0,inclass[i])
            sheetf2.write(count2,1,intag[i])
            sheetf2.write(count2,2,inwords[i])
            count2+=1
        else:
            sheetf1.write(count1,0,inclass[i])
            sheetf1.write(count1,1,intag[i])
            sheetf1.write(count1,2,inwords[i])
            count1=count1+1
    sheetf2.write(0,4,"总数")
    sheetf1.write(0,4,"总数")
    sheetf2.write(1,4,count2)
    sheetf1.write(1,4,count1)
    f1.save("combine_train.xls")
    f2.save("combine_test.xls")


def step2_pre_processing(suffix_stop="",suffix_start="",suffix_fre=""):
    r=xlrd.open_workbook(filename="combine_"+suffix_start+".xls")
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
            if word_freqs[i]>0:
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
    w.save("pre处理"+suffix_stop+suffix_start+suffix_fre+".xls")
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
    ex=0
    ex_tag=[]
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
                ex_tag.append(ex)
                swich=1
            else:
                mid=precess_words[0:maxlen]
                out2_words.append(mid)
                out2_tag.append(intag[i])
                out2_class.append(inclass[i])
                ex_tag.append(ex)
            precess_words=precess_words[maxlen:len(precess_words)].strip()
            
        if len(precess_words)!=0:
            if swich==0:
                out1_words.append(precess_words)
                out2_words.append(precess_words)
                out1_tag.append(intag[i])
                out2_tag.append(intag[i])
                out1_class.append(inclass[i])
                out2_class.append(inclass[i])
                ex_tag.append(ex)
            elif len(precess_words)>maxlen/4:
                out2_words.append(precess_words)
                out2_tag.append(intag[i])
                out2_class.append(inclass[i])
                ex_tag.append(ex)
        ex+=1
    print("start save.........")
    w=xlwt.Workbook()
    sheet2=w.add_sheet("准备文件",cell_overwrite_ok=True)
    sheet2.write(0,14,maxlen)
    sheet2.write(0,11,"切割丢掉")
    sheet2.write(1,11,len(out1_tag))
    sheet2.write(0,12,"切割扩充")
    sheet2.write(1,12,len(out2_tag))
    swich=0
    for i in range(0,len(out2_words)):
        if i==65000:
            swich+=1
            break
        sheet2.write(i,3,out2_class[i])
        sheet2.write(i,4,out2_tag[i])
        sheet2.write(i,5,out2_words[i])
        sheet2.write(i,6,ex_tag[i])
    if swich==1:
        sheet2.write(0,13,1)
        for i in range(65000,len(out2_words)):
            sheet2.write(i-65000,7,out2_class[i])
            sheet2.write(i-65000,8,out2_tag[i])
            sheet2.write(i-65000,9,out2_words[i])
            sheet2.write(i-65000,10,ex_tag[i])
    else:
        sheet2.write(0,13,0)
    for i in range(0,len(out1_words)):
        sheet2.write(i,0,out1_class[i])
        sheet2.write(i,1,out1_tag[i])
        sheet2.write(i,2,out1_words[i])
    w.save("切割"+suffix+".xls")
def model(suffix="",suffix_fre=""):
    for round in range(0,2):
        rtest=xlrd.open_workbook(filename="切割"+suffix+"test"+suffix_fre+".xls")
        rtrain=xlrd.open_workbook(filename="切割"+suffix+"train"+suffix_fre+".xls")
        r_vocall1=xlrd.open_workbook(filename="pre处理"+suffix+"test"+suffix_fre+".xls")
        r_vocall2=xlrd.open_workbook(filename="pre处理"+suffix+"train"+suffix_fre+".xls")
        sheet_test=rtest.sheet_by_index(0)
        sheet_train=rtrain.sheet_by_index(0)
        sheet1_vocall=r_vocall1.sheet_by_index(0)
        sheet2_vocall=r_vocall2.sheet_by_index(0)
        invocal1=sheet1_vocall.col_values(4)
        
        invocal2=sheet2_vocall.col_values(4)
        for i in range(0,len(invocal1)):
            if len(invocal1[i])==0:
                invocall=invocal1[:i]
                print("1")
                break
            if i==len(invocal1)-1:
                invocall=invocal1
        
        for i in range(0,len(invocal2)):
            if len(invocal2[i])==0:
                print("1")
                invocal2=invocal2[:i]
                break
        for i in invocal2:
            if i not in invocall:
                invocall.append(i)
        print(len(invocall))
        vocall_size=len(invocall)
        if round==1:
            ex_tag=sheet_test.col_values(6)
        xtrain=sheet_train.col_values(2+round*3)
        ztrain=sheet_train.col_values(0+round*3)
        ytrain=sheet_train.col_values(1+round*3)
        xtest=sheet_test.col_values(2+round*3)
        ztest=sheet_test.col_values(0+round*3)
        ytest=sheet_test.col_values(1+round*3)
        
        for i in range(0,len(xtrain)):
            if len(xtrain[i])==0:
                xtrain=xtrain[:i]
                ztrain=ztrain[:i]
                ytrain=ytrain[:i]
                break
        for i in range(0,len(xtest)):
            if len(xtest[i])==0:
                xtest=xtest[:i]
                ytest=ytest[:i]
                ztest=ztest[:i]
                break
        print(round*3)
        print(len(xtrain),"xtrain")
        print(len(ytrain),"ytrain")
        print(len(xtest),"xtest")
        print(len(ytest),"ytest")
        if round==1:
            other=sheet_train.cell(0,13).value
            other=int(other)
            print(other)
            if other==1:
                xtrain=xtrain+sheet_train.col_values(9)
                ytrain=ytrain+sheet_train.col_values(8)
                ztrain=ztrain+sheet_train.col_values(7)
                for i in range(0,len(xtrain)):
                    if len(xtrain[i])==0:
                        xtrain=xtrain[:i]
                        ztrain=ztrain[:i]
                        ytrain=ytrain[:i]
                        break

        tokenizer=Tokenizer(num_words=vocall_size)
        tokenizer.fit_on_texts(invocall)
        xtrain=tokenizer.texts_to_sequences(xtrain)
        xtest=tokenizer.texts_to_sequences(xtest)
        maxlen=0
        for i in xtrain:
            if len(i)>maxlen:
                maxlen=len(i)
        for i in xtest:
            if len(i)>maxlen:
                maxlen=len(i)
        print(maxlen,"maxlen")
        xtrain=pad_sequences(xtrain,padding='post',maxlen=maxlen)
        xtest=pad_sequences(xtest,padding='post',maxlen=maxlen)
        print(len(ytrain),len(xtrain))
        print(len(ytest),len(xtest))
        for i in range(0,len(ytrain)):
            ytrain[i]=int(ytrain[i])
        for i in range(0,len(ytest)):
            ytest[i]=int(ytest[i])
        embedding_size=150
        hidden_layer_size=64
        batch_size=128
        num_epochs=3
        model=Sequential()
        model.add(Embedding(vocall_size,embedding_size,input_length=maxlen))
        model.add(AT(25,150))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(hidden_layer_size,dropout=0.2,recurrent_dropout=0.2))
        
        model.add(Dense(10))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        model.summary()
        
        history=model.fit(xtrain,ytrain,epochs=7,batch_size=64)
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
        sheet2.write(0,8,"predict")
        sheet2.write(0,9,"ytest")
        sheet2.write(0,10,"xtest")
        sheet2.write(0,11,"ex_tag")
        sheet2.write(0,4,"loss")
        sheet2.write(1,4,loss)
        sheet2.write(0,5,"acc")
        sheet2.write(1,5,accuracy)
        ypred=model.predict_classes(xtest,1)
        xtest=tokenizer.sequences_to_texts(xtest)
        for index in range(0,len(ypred)):
            sheet2.write(index,0,int(ypred[index][0]))
            sheet2.write(index,1,ytest[index])
            sheet2.write(index,2,xtest[index])
            if round==1:
                sheet2.write(index,3,ex_tag[index])
      
        if round==0:
            w.save("result切割"+suffix+suffix_fre+"at.xls")
        else:
            w.save("result扩充"+suffix+suffix_fre+"at.xls")
def acc(suffix="",suffix2="",suffix_fre=""):
    r=xlrd.open_workbook(filename="result扩充"+suffix+suffix_fre+suffix2+".xls")
    sheet1=r.sheet_by_index(0)
    ytest=sheet1.col_values(1)
    ypred=sheet1.col_values(0)
    ex_tag=sheet1.col_values(3)
    print(len(ytest))
    print(len(ypred))
    print(len(ex_tag))
    sum_count=0.0
    acc_sum=0.0
    exp=ex_tag[0]
    count=0.0
    tag_sum_pre=0.0
    tag_sum=0.0
    i=0
    while(i<len(ex_tag)):
        if ex_tag[i]==exp:
            count=count+1
            tag_sum_pre=tag_sum_pre+float(ypred[i])
            tag_sum=float(ytest[i])+tag_sum
        else:
            exp=ex_tag[i]
            if count==0:
                print(count,"!!")
            if tag_sum_pre/count>=0.5:
                if tag_sum/count==1:
                    acc_sum+=1
            else:
                if tag_sum==0:
                    acc_sum+=1
            count=0.0
            tag_sum_pre=0.0
            tag_sum=0.0
            i-=1
            sum_count+=1
        i+=1
    print(acc_sum)
    print(sum_count)
    acr=acc_sum/sum_count
    w=xlwt.Workbook()
    sheet2=w.add_sheet("准备文件",cell_overwrite_ok=True)
    sheet2.write(0,7,"acc")
    sheet2.write(1,7,acr)
    w.save("acc"+suffix+suffix2+suffix_fre+".xls")

def model_word2vec(suffix="",suffix_fre=""):
    for round in range(1,2):
        rtest=xlrd.open_workbook(filename="切割"+suffix+"test"+suffix_fre+".xls")
        rtrain=xlrd.open_workbook(filename="切割"+suffix+"train"+suffix_fre+".xls")
        r_vocall1=xlrd.open_workbook(filename="pre处理"+suffix+"test"+suffix_fre+".xls")
        r_vocall2=xlrd.open_workbook(filename="pre处理"+suffix+"train"+suffix_fre+".xls")
        sheet_test=rtest.sheet_by_index(0)
        sheet_train=rtrain.sheet_by_index(0)
        sheet1_vocall=r_vocall1.sheet_by_index(0)
        sheet2_vocall=r_vocall2.sheet_by_index(0)
        invocal1=sheet1_vocall.col_values(4)
        
        invocal2=sheet2_vocall.col_values(4)
        for i in range(0,len(invocal1)):
            if len(invocal1[i])==0:
                invocall=invocal1[:i]
                print("1")
                break
            if i==len(invocal1)-1:
                invocall=invocal1
        
        for i in range(0,len(invocal2)):
            if len(invocal2[i])==0:
                print("1")
                invocal2=invocal2[:i]
                break
        for i in invocal2:
            if i not in invocall:
                invocall.append(i)
        print(len(invocall))
        vocall_size=len(invocall)
        if round==1:
            ex_tag=sheet_test.col_values(6)
        xtrain=sheet_train.col_values(2+round*3)
        ztrain=sheet_train.col_values(0+round*3)
        ytrain=sheet_train.col_values(1+round*3)
        xtest=sheet_test.col_values(2+round*3)
        ztest=sheet_test.col_values(0+round*3)
        ytest=sheet_test.col_values(1+round*3)
        
        for i in range(0,len(xtrain)):
            if len(xtrain[i])==0:
                xtrain=xtrain[:i]
                ztrain=ztrain[:i]
                ytrain=ytrain[:i]
                break
        for i in range(0,len(xtest)):
            if len(xtest[i])==0:
                xtest=xtest[:i]
                ytest=ytest[:i]
                ztest=ztest[:i]
                break
        print(round*3)
        print(len(xtrain),"xtrain")
        print(len(ytrain),"ytrain")
        print(len(xtest),"xtest")
        print(len(ytest),"ytest")
        if round==1:
            other=sheet_train.cell(0,13).value
            other=int(other)
            print(other)
            if other==1:
                xtrain=xtrain+sheet_train.col_values(9)
                ytrain=ytrain+sheet_train.col_values(8)
                ztrain=ztrain+sheet_train.col_values(7)
            for i in range(0,len(xtrain)):
                if len(xtrain[i])==0:
                    xtrain=xtrain[:i]
                    ztrain=ztrain[:i]
                    ytrain=ytrain[:i]
                    break

        tokenizer=Tokenizer(num_words=vocall_size)
        tokenizer.fit_on_texts(invocall)
        xtrain=tokenizer.texts_to_sequences(xtrain)
        xtest=tokenizer.texts_to_sequences(xtest)
        maxlen=0
        for i in xtrain:
            if len(i)>maxlen:
                maxlen=len(i)
        for i in xtest:
            if len(i)>maxlen:
                maxlen=len(i)
        print(maxlen,"maxlen")
        xtrain=pad_sequences(xtrain,padding='post',maxlen=maxlen)
        xtest=pad_sequences(xtest,padding='post',maxlen=maxlen)
        print(len(ytrain),len(xtrain))
        print(len(ytest),len(xtest))
        for i in range(0,len(ytrain)):
            ytrain[i]=int(ytrain[i])
        for i in range(0,len(ytest)):
            ytest[i]=int(ytest[i])
        modelw2v = gensim.models.KeyedVectors.load("word2vec_150_lstm.model")
        embedding_matrix = np.zeros(shape=(vocall_size ,150))
        for i,word in enumerate(invocall):
            try:
                embedding_vector = modelw2v[word]
                embedding_matrix[i,:] = embedding_vector
            except KeyError:
                pass
        embedding_size=150
        hidden_layer_size=64
        batch_size=128
        num_epochs=3
        model=Sequential()
        model.add(Embedding(vocall_size,embedding_size,weights=[embedding_matrix],input_length=maxlen))
        model.add(AT(25,150))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(hidden_layer_size,dropout=0.2,recurrent_dropout=0.2))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        model.summary()
        
        history=model.fit(xtrain,ytrain,epochs=7,batch_size=64)
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
        sheet2.write(0,8,"predict")
        sheet2.write(0,9,"ytest")
        sheet2.write(0,10,"xtest")
        sheet2.write(0,11,"ex_tag")
        sheet2.write(0,4,"loss")
        sheet2.write(1,4,loss)
        sheet2.write(0,5,"acc")
        sheet2.write(1,5,accuracy)
        ypred=model.predict_classes(xtest,1)
        xtest=tokenizer.sequences_to_texts(xtest)
        for index in range(0,len(ypred)):
            sheet2.write(index,0,int(ypred[index][0]))
            sheet2.write(index,1,ytest[index])
            sheet2.write(index,2,xtest[index])
            if round==1:
                sheet2.write(index,3,ex_tag[index])
        if round==0:
            w.save("result切割"+suffix+suffix_fre+"w2v_at.xls")
        else:
            w.save("result扩充"+suffix+suffix_fre+"w2v_at.xls")
"""
step2_pre_processing(suffix_stop="3",suffix_start="train",suffix_fre="off")
step2_pre_processing(suffix_stop="3",suffix_start="test",suffix_fre="off")
spilt_words(suffix="3trainoff")
spilt_words(suffix="3testoff")
step2_pre_processing(suffix_stop="2",suffix_start="train",suffix_fre="off")
step2_pre_processing(suffix_stop="2",suffix_start="test",suffix_fre="off")
spilt_words(suffix="2trainoff")
spilt_words(suffix="2testoff")
step2_pre_processing(suffix_stop="1",suffix_start="train",suffix_fre="off")
step2_pre_processing(suffix_stop="1",suffix_start="test",suffix_fre="off")
spilt_words(suffix="1trainoff")
spilt_words(suffix="1testoff")
"""


model("3","off")



model_word2vec("3","off")



acc("3",suffix_fre="off")



acc("3","w2v",suffix_fre="off")


      



