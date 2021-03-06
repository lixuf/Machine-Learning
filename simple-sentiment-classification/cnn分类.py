from keras.layers.core import Activation,Dense,Dropout,SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Conv1D
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

def model(suffix="",suffix_fre=""):
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
    class_num=sheet2_vocall.col_values(10)
    allclass=[]
    for i in range(1,len(class_num)):
        if class_num[i]!="":
            if class_num[i] not in allclass:
                allclass.append(class_num[i])
    print(allclass)

    for all_round in range(0,len(allclass)):
        for round in range(0,2):

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
            print(len(ztrain),"ztrain")
            print(len(xtest),"xtest")
            print(len(ztest),"ztest")
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
            print("class",allclass)
            for i in range(0,len(ztest)):
                for n1 in range(0,len(allclass)):
                    if ztest[i]==allclass[n1]:
                        ztest[i]=n1
            for i in range(0,len(ztrain)):
                for n1 in range(0,len(allclass)):
                    if ztrain[i]==allclass[n1]:
                        ztrain[i]=n1
                        
            xtrain=pad_sequences(xtrain,padding='post',maxlen=maxlen)
            xtest=pad_sequences(xtest,padding='post',maxlen=maxlen)
            print(len(ztrain),len(xtrain))
            print(len(ztest),len(xtest))
            for i in range(0,len(ztrain)):
               if ztrain[i]="":
                   ztrain
                    ztrain[i]=int(ztrain[i])
               
            for i in range(0,len(ztest)):
                ztest[i]=int(ztest[i])
            embedding_size=150
            hidden_layer_size=64
            batch_size=128
            num_epochs=3
            model=Sequential()
            model.add(Embedding(vocall_size,embedding_size,input_length=maxlen))
            model.add(SpatialDropout1D(0.2))
            model.add(LSTM(hidden_layer_size,dropout=0.2,recurrent_dropout=0.2))
            model.add(Dense(1))
            model.add(Activation("sigmoid"))
            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
            model.summary()
        
            history=model.fit(xtrain,ztrain,epochs=2,batch_size=64)
            loss,accuracy=model.evaluate(xtest,ztest)
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
            sheet2.write(0,9,"ztest")
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
                sheet2.write(index,1,ztest[index])
                sheet2.write(index,2,xtest[index])
                if round==1:
                    sheet2.write(index,3,ex_tag[index])
      
            if round==0:
                w.save("result切割"+suffix+allclass[all_round]+suffix_fre+".xls")
            else:
                w.save("result扩充"+suffix+allclass[all_round]+suffix_fre+".xls")
    return allclass
def acc(suffix="",suffix2="",all_class=[],suffix_fre=""):
    for class_tag in all_class:
        r=xlrd.open_workbook(filename="result扩充"+suffix+class_tag+suffix_fre+suffix2+".xls")
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
        w.save("acc"+suffix+suffix2+class_tag+suffix_fre+".xls")

def model_word2vec(suffix="",suffix_fre=""):
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
    class_num=sheet2_vocall.col_values(10)
    allclass=[]
    for i in range(1,len(class_num)):
        if class_num[i]!="":
            if class_num[i] not in allclass:
                allclass.append(class_num[i])
    print(allclass)

    for all_round in range(0,len(allclass)):
        for round in range(0,2):

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
            print(len(ztrain),"ztrain")
            print(len(xtest),"xtest")
            print(len(ztest),"ztest")
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
            for i in range(0,len(ztest)):
                for n1 in range(0,len(allclass)):
                    if ztest[i]==allclass[n1]:
                        ztest[i]=n1
            for i in range(0,len(ztrain)):
                for n1 in range(0,len(allclass)):
                    if ztrain[i]==allclass[n1]:
                        ztrain[i]=n1
            xtrain=pad_sequences(xtrain,padding='post',maxlen=maxlen)
            xtest=pad_sequences(xtest,padding='post',maxlen=maxlen)
            print(len(ztrain),len(xtrain))
            print(len(ztest),len(xtest))
            for i in range(0,len(ztrain)):
                ztrain[i]=int(ztrain[i])
            for i in range(0,len(ztest)):
                ztest[i]=int(ztest[i])
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
            model.add(SpatialDropout1D(0.2))
            model.add(LSTM(hidden_layer_size,dropout=0.2,recurrent_dropout=0.2))
            model.add(Dense(1))
            model.add(Activation("sigmoid"))
            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
            model.summary()
        
            history=model.fit(xtrain,ztrain,epochs=1,batch_size=64)
            loss,accuracy=model.evaluate(xtest,ztest)
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
            sheet2.write(0,9,"ztest")
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
                sheet2.write(index,1,ztest[index])
                sheet2.write(index,2,xtest[index])
                if round==1:
                    sheet2.write(index,3,ex_tag[index])
            if round==0:
                w.save("result切割"+suffix+allclass[all_round]+suffix_fre+"w2v.xls")
            else:
                w.save("result扩充"+suffix+allclass[all_round]+suffix_fre+"w2v.xls")
    return allclass

"""
step2_pre_processing(suffix_stop="3",suffix_start="train")
step2_pre_processing(suffix_stop="3",suffix_start="test")
spilt_words(suffix="3train")
spilt_words(suffix="3test")
"""
"""
all_class=model("2")
acc("2",all_class=all_class)
"""
"""
all_class=model_word2vec("2")
acc("2","w2v",all_class=all_class)

all_class=model("1")
acc("1",all_class=all_class)

all_class=model_word2vec("1")
acc("1","w2v",all_class=all_class)
"""
all_class=model("3")
acc("3",all_class=all_class)

all_class=model_word2vec("3")
acc("3","w2v",all_class=all_class)


"""
all_class=model("2","off")
acc("2",all_class=all_class,suffix_fre="off")

all_class=model_word2vec("2",suffix_fre="off")
acc("2","w2v",all_class=all_class,suffix_fre="off")

all_class=model("1",suffix_fre="off")
acc("1",all_class=all_class,suffix_fre="off")

all_class=model_word2vec("1",suffix_fre="off")
acc("1","w2v",all_class=all_class,suffix_fre="off")
"""

all_class=model("3",suffix_fre="off")
acc("3",all_class=all_class,suffix_fre="off")

all_class=model_word2vec("3",suffix_fre="off")
acc("3","w2v",all_class=all_class,suffix_fre="off")