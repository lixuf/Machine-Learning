from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.callbacks import Callback
import numpy as np
import json
from keras.preprocessing.text import text_to_word_sequence as sq2wsq
import numpy as np
import os
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
import multiprocessing
from data import gen_data
from model import ExponentialMovingAverage


def dilated_gated_conv1d(seq, dilation_rate=1):
        """膨胀门卷积（残差式）
        """
        dim= K.int_shape(seq)[-1]
        h = Conv1D(dim*2, 3, padding='same', dilation_rate=dilation_rate)(seq)
        def _gate(x):
            dropout_rate = 0.1
            s, h = x
            g, h = h[:, :, :dim], h[:, :, dim:]
            g = K.in_train_phase(K.dropout(g, dropout_rate), g)
            g = K.sigmoid(g)
            return g * s + (1 - g) * h
        seq = Lambda(_gate)([seq, h])
        return seq

filelist=[]
fend1=2
fend2=2
for i in range(1,fend1):
    for i1 in range(1,fend2):
        if i ==1:
            if i1==4:
                break
        filelist.append("phaseB_"+str(i)+"b_0"+str(i1)+".json")

train_D=gen_data(filelist)
data=train_D.data
N=train_D.N
tag=train_D.tag
print("type(tag)",type(tag))
word_size=train_D.word_size


t_in=Input(shape=(None,word_size*2))
tag_in=Input(shape=(None,))
n1_in=Input(shape=(None,))
n2_in=Input(shape=(None,))
t,tagm,n1,n2=t_in,tag_in,n1_in,n2_in

td=dilated_gated_conv1d(seq=t, dilation_rate=1)
td=dilated_gated_conv1d(seq=td, dilation_rate=2)
td=dilated_gated_conv1d(seq=td, dilation_rate=5)
td=dilated_gated_conv1d(seq=td, dilation_rate=1)
encoder=Dense(word_size)(td)
encoder_model=Model(t_in,encoder)

decoder=Dense(word_size*2)(encoder)
dc=dilated_gated_conv1d(seq=decoder, dilation_rate=1)
dc=dilated_gated_conv1d(seq=dc, dilation_rate=5)
dc=dilated_gated_conv1d(seq=dc, dilation_rate=2)
dc=dilated_gated_conv1d(seq=dc, dilation_rate=1)
o1m=Lambda(lambda x : x[:,:,:word_size])(dc)
o2m=Lambda(lambda x : x[:,:,word_size:])(dc)
decoder_model=Model([t_in],[o1m,o2m])

cata=dilated_gated_conv1d(seq=encoder, dilation_rate=1)
cata=Lambda(lambda x : K.sum(x,axis=0))(cata)
cata=Dense(2,activation='softmax')(cata)
cata_model=Model(t_in,cata)

def mean_squared_error(y_true, y_pred):
            if not K.is_tensor(y_pred):
                y_pred = K.constant(y_pred)
            y_true = K.cast(y_true, y_pred.dtype)
            return K.mean(K.square(y_pred - y_true), axis=-1)
t1=Lambda(lambda x : x[:,:,:word_size])(t)
t2=Lambda(lambda x : x[:,:,word_size:])(t)
o1_loss=(n1/(n1+n2))*(mean_squared_error(t1,o1m))
o2_loss=(n2/(n1+n2))*(mean_squared_error(t2,o2m))
c_loss=K.categorical_crossentropy(tagm,cata)
loss=(o1_loss+o2_loss)+(c_loss)

rae_model=Model([t_in,tag_in,n1_in,n2_in],[encoder,o1m,o2m,cata])
rae_model.add_loss(loss)
rae_model.compile(optimizer=Adam(1e-3))
rae_model.summary()

def t():
        def ju_juli(indata,o1,o2,n1,n2,tag):# 0相似 1不相似
                def npsquared_error(y_true, y_pred):
                    y_true=np.array(y_true)
                    y_pred=np.array(y_pred)
                    return np.mean(np.abs(y_pred - y_true), axis=-1)
                print(type(o1))
                print(type(indata))
                o1_loss=(n1/(n1+n2))*(npsquared_error(indata[:,:,:word_size],o1))
                o2_loss=(n2/(n1+n2))*(npsquared_error(indata[:,:,word_size:],o2))
                print("o1loss",type(o1_loss))
                o=o1_loss+o2_loss
                print("type o ",type(o))
                o=o[:,:1]

                otag=cata_model.predict(indata)
                otag=np.where(otag>=0.5,1,0)
                retu1=np.where(o<0.05,True,False)

                retu2=np.where(otag==tag,True,False)
                retu2=retu2[:,:1]

                return retu1*retu2
        chulin=2
        rae_model.load_weights('rae_model.weights')
        encoder_model.load_weights("encoder_model.weights")
        decoder_model.load_weights("decoder_model.weights")
        cata_model.load_weights("cata_model.weights")
        erase=[]
        for i1,i2 in enumerate(data):
            if len(i2)==1:
                self.result["data"].append(i2)
                self.result["tag"].append(tag[i1])
                erase.append(i1)
                continue
            Nnew=[]
            datanew=[]
            i3=0
            indata=[]
            inN1=[]
            inN2=[]
            intag=[]
            print("data "+str(i1),len(i2))
            while i3+1<len(i2):
                t1=np.array(i2[i3]).reshape(1,word_size)
                t2=np.array(i2[i3+1]).reshape(1,word_size)
                indata.append(np.concatenate([t1,t2],-1))
                inN1.append(N[i1][i3])
                inN2.append(N[i1][i3+1])
                intag.append(tag[i1])
                i3=i3+1
            print("indata ",len(indata))
            indata=np.array(indata)
            inN1=np.array(inN1)
            inN2=np.array(inN2)
            intag=np.array(intag)
            o1,o2=decoder_model.predict(indata)
            tiao=0
            encoderdata=[]
            encoderdata=encoder_model.predict(indata)
            for i3,ex in enumerate(ju_juli(indata,o1,o2,inN1,inN2,intag)):
                print("ex",ex)
                print("tiao",tiao)
                if tiao==1:
                    tiao=0
                    continue
                if ex:
                    datanew.append(encoderdata[i3])
                    Nnew.append(N[i1][i3]+N[i1][i3+1])
                    tiao=1
                else:
                    datanew.append(i2[i3])
                    Nnew.append(N[i1][i3])
                    if i3+2==len(i2):
                        datanew.append(i2[i3+1])
                        Nnew.append(N[i1][i3+1])
            if len(data[i1])-len(datanew)*2==1:
                Nnew.append(N[i1][-1])
                datanew.append(data[i1][-1])    
            print("shape datanew",np.shape(datanew))
            data[i1]=datanew
            print(Nnew,np.shape(Nnew))
            N[i1]=Nnew
       
        for i in erase:
            del data[i]
            del N[i]
        if len(erase)/len(data)>chulin:
            chulin=len(erase)/len(data)
            print("处理个数",chulin)
            rae_model.save_weights('best_model.weights')

  
        

t()