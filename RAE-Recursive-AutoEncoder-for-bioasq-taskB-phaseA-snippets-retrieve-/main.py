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
import copy

class Evaluate(Callback):
    def __init__(self):
        self.chulin=0.01
        self.best = 0.
        self.passed = 0
        self.stage = 0
        self.result={}
        self.result["data"]=[]
        self.result["tag"]=[]

    def on_epoch_end(self, epoch, logs=None):

        EMAer.apply_ema_weights()
        #data N decoder_model encoder_model tag
        erase=[]
        print("old data[11] len",len(datae[11]))
        for i1,i2 in enumerate(datae):
            if len(i2)==1:
                self.result["data"].append(i2)
                self.result["tag"].append(tage[i1])
                erase.append(i1)
                continue
            Nnew=[]
            datanew=[]
            i3=0
            indata=[]
            inN1=[]
            inN2=[]
            intag=[]
            
            while i3+1<len(i2):
                t1=np.array(i2[i3]).reshape(1,word_size)
                t2=np.array(i2[i3+1]).reshape(1,word_size)
                indata.append(np.concatenate([t1,t2],-1))
                inN1.append(Ne[i1][i3])
                inN2.append(Ne[i1][i3+1])
                intag.append(tage[i1])
                i3=i3+1
            indata=np.array(indata)
            inN1=np.array(inN1)
            inN2=np.array(inN2)
            intag=np.array(intag)
            o1,o2=decoder_model.predict(indata)
            tiao=0
            encoderdata=[]
            encoderdata=encoder_model.predict(indata)
            for i3,ex in enumerate(self.ju_juli(indata,o1,o2,inN1,inN2,intag,i1)):
                if tiao==1:
                    tiao=0
                    continue
                if ex:
                    datanew.append(encoderdata[i3])
                    Nnew.append(Ne[i1][i3]+Ne[i1][i3+1])
                    tiao=1
                else:
                    datanew.append(i2[i3])
                    Nnew.append(Ne[i1][i3])
                    if i3+2==len(i2):
                        datanew.append(i2[i3+1])
                        Nnew.append(Ne[i1][i3+1])
            if len(datae[i1])-len(datanew)*2==1:
                Nnew.append(Ne[i1][-1])
                datanew.append(datae[i1][-1]) 
            datae[i1]=datanew
            Ne[i1]=Nnew
        erase=sorted(erase)
        for qt,i in enumerate(erase):
            del datae[i-qt]
            del Ne[i-qt]
        if len(erase)/len(datae)>self.chulin:
            self.chulin=len(erase)/len(datae)
            print("处理个数",self.chulin)
        train_D.updata(datae,Ne,tage)
        EMAer.reset_old_weights()
        
        
    def ju_juli(self,indata,o1,o2,n1,n2,tagj,i):# 0相似 1不相似
        def np_error(y_truee, y_prede):
            y_truee=np.array(y_truee)
            y_prede=np.array(y_prede)
            return np.mean(np.abs(y_prede - y_truee), axis=-1)
        o1_losse=(n1/(n1+n2))*(np_error(indata[:,:,:word_size],o1))
        o2_losse=(n2/(n1+n2))*(np_error(indata[:,:,word_size:],o2))
        o=o1_losse+o2_losse
        o=o[:,:1]
        otag=cata_model.predict(indata)
        otag=np.where(otag>=0.5,1,0)
        retu1=np.where(o<0.1,True,False)
        retu2=np.where(otag==tagj,True,False)
        retu2=retu2[:,:1]
        if retu2[0]==False:
            print(i,"!!!!!!")
            print("o",o)
            print("otag",otag)
            print("tagj",tagj)
            print("rete1",retu1)
            print("rete2",retu2)
        return retu1*retu2
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
fend1=3
fend2=2
for i in range(1,fend1):
    for i1 in range(1,fend2):
        if i ==1:
            if i1==4:
                break
        filelist.append("phaseB_"+str(i)+"b_0"+str(i1)+".json")
filelist.append("phaseB_1b_01F.json")

train_D=gen_data(filelist)
datae=copy.deepcopy(train_D.data)
Ne=copy.deepcopy(train_D.N)
tage=copy.deepcopy(train_D.tag)
word_size=train_D.word_size

t_in=Input(shape=(None,word_size*2))
tag_in=Input(shape=(None,))
n1_in=Input(shape=(None,))
n2_in=Input(shape=(None,))
t,tagm,n1,n2=t_in,tag_in,n1_in,n2_in

td=dilated_gated_conv1d(t,1)
td=dilated_gated_conv1d(td,2)
td=dilated_gated_conv1d(td,5)
td=dilated_gated_conv1d(td,1)
encoder=Dense(word_size)(td)
encoder_model=Model(t_in,encoder)

decoder=Dense(word_size*2)(encoder)
dc=dilated_gated_conv1d(decoder,1)
dc=dilated_gated_conv1d(dc,2)
dc=dilated_gated_conv1d(dc,5)
dc=dilated_gated_conv1d(dc,1)
o1m=Lambda(lambda x : x[:,:,:word_size])(dc)
o2m=Lambda(lambda x : x[:,:,word_size:])(dc)
decoder_model=Model([t_in],[o1m,o2m])

cata=Conv1D(word_size,1, activation='relu')(encoder)
cata=Conv1D(word_size,1, activation='relu')(cata)
cata=Conv1D(word_size,1, activation='relu')(cata)
cata=Conv1D(word_size,1, activation='relu')(cata)
cata=Conv1D(word_size,1, activation='relu')(cata)
cata=Conv1D(word_size,1, activation='relu')(cata)
cata=GlobalMaxPooling1D()(cata)
cata=Dense(2,activation='softmax')(cata)
cata_model=Model(t_in,cata)

def mean_squared_error(y_true, y_pred):
            if not K.is_tensor(y_pred):
                y_pred = K.constant(y_pred)
            y_true = K.cast(y_true, y_pred.dtype)
            return K.mean(K.square(y_pred - y_true), axis=-1)
t1m=Lambda(lambda x : x[:,:,:word_size])(t)
t2m=Lambda(lambda x : x[:,:,word_size:])(t)
o1_loss=(n1/(n1+n2))*(mean_squared_error(t1m,o1m))
o2_loss=(n2/(n1+n2))*(mean_squared_error(t2m,o2m))
c_loss=K.binary_crossentropy(tagm,cata)
loss=(o1_loss+o2_loss)+(c_loss)

rae_model=Model([t_in,tag_in,n1_in,n2_in],[encoder,o1m,o2m,cata])
rae_model.add_loss(loss)
rae_model.compile(optimizer=Adam(1e-3))
rae_model.summary()

EMAer = ExponentialMovingAverage(rae_model)
EMAer.inject()
evaluator=Evaluate()

ep=150
now=0
while now<ep and len(train_D)>10:
      if now!=0:
          rae_model.load_weights('rae_model.weights')
      thistory=rae_model.fit_generator(train_D.__iter__(),
                                      steps_per_epoch=len(train_D),
                                      epochs=1,
                                      callbacks=[evaluator]
                                      )
      
      tlosss=thistory.history()["loss"]
      rae_model.save_weights('rae_model.weights')
      print(now)
      now=now+1
                              
      
      
rae_model.save_weights('rae_model.weights')
encoder_model.save_weights("encoder_model.weights")
decoder_model.save_weights("decoder_model.weights")
cata_model.save_weights("cata_model.weights")