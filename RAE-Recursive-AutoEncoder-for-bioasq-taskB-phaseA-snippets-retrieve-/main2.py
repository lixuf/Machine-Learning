
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
        self.gat=0.2
    def on_epoch_end(self, epoch, logs=None):

 
        #data N decoder_model encoder_model tag
        erase=[]
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
            if len(datanew)==1:
                self.result["data"].append(datanew)
                self.result["tag"].append(Nnew)
                erase.append(i1)
            else:
                datae[i1]=datanew
                Ne[i1]=Nnew
        erase=sorted(erase)
        for qt,i in enumerate(erase):
            del datae[i-qt]
            del Ne[i-qt]
            del tage[i-qt]

        train_D.updata(datae,Ne,shujutag=tage)
        print("data",len(self.result["data"]))
        print("tag",len(self.result["tag"]))        
        
    def ju_juli(self,indata,o1,o2,n1,n2,tagj,i):# 0相似 1不相似
        def np_error(y_truee, y_prede):
            y_truee=np.array(y_truee)
            y_prede=np.array(y_prede)
            return np.sqrt(np.sum(np.square(y_prede-y_truee),axis=-1))
        o1_losse=(n1/(n1+n2))*(np_error(indata[:,:,:word_size],o1))
        o2_losse=(n2/(n1+n2))*(np_error(indata[:,:,word_size:],o2))
        o=o1_losse+o2_losse
        o=o[:,:1]
        otag=cata_model.predict(indata)
        otag=np.where(otag>=0.5,1,0)
        retu1=np.where(o<self.gat,True,False)       
        tagj=np.array(tagj).reshape(np.shape(otag))
        retu2=np.where(otag==tagj,True,False)
        retu2=retu2[:,:1]
        if i==2 or i==2000:
            print(i,"!!!!!!")
            print("o",o)
            print("otag",otag)
            print("tagj",tagj)
            print("rete1",retu1)
            print("rete2",retu2)
        return retu1*retu2
    def updata_increa(self,pp):
        self.gat=self.gat+pp
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
fend1=6
fend2=6
for i in range(1,fend1):
    for i1 in range(1,fend2):
        if i ==1:
            if i1==4:
                break
        filelist.append("phaseB_"+str(i)+"b_0"+str(i1)+".json")
        if i==4 or i==5:
            filelist.append("phaseA_"+str(i)+"b_0"+str(i1)+"D.json")
for i in range(2,9):
    filelist.append("BioASQ-trainingDataset"+str(i)+"b.json")

filelist.append("phaseA_4b_01T.json")
filelist.append("phaseA_4b_02T.json")


train_D=gen_data(filelist)
datae=copy.deepcopy(train_D.data)
Ne=copy.deepcopy(train_D.N)
tage=copy.deepcopy(train_D.tag)
word_size=train_D.word_size
print("原始数据：：：",len(datae))
t_in=Input(shape=(None,word_size*2))
tag_in=Input(shape=(None,))
n1_in=Input(shape=(None,))
n2_in=Input(shape=(None,))
t,tagm,n1,n2=t_in,tag_in,n1_in,n2_in

td=dilated_gated_conv1d(t,1)
td=dilated_gated_conv1d(td,1)
td=dilated_gated_conv1d(td,1)
td=dilated_gated_conv1d(td,1)
encoder=Dense(word_size)(td)
encoder=Lambda(lambda x : x /K.expand_dims(K.sqrt (K.sum (K.square(x),axis=-1) ),axis=-2))(encoder)
encoder_model=Model(t_in,encoder)

decoder=Dense(word_size*2)(encoder)
dc=dilated_gated_conv1d(decoder,1)
dc=dilated_gated_conv1d(dc,1)
dc=dilated_gated_conv1d(dc,1)
dc=dilated_gated_conv1d(dc,1)
o1m=Lambda((lambda x : x[:,:,:word_size]),name="dc1")(dc)
o2m=Lambda((lambda x : x[:,:,word_size:]),name="dc2")(dc)
decoder_model=Model([t_in],[o1m,o2m])

encoder_c=encoder
cata=Conv1D(word_size,1, activation='relu')(encoder_c)
cata=Conv1D(word_size,1, activation='relu')(cata)
cata=Conv1D(word_size,1, activation='relu')(cata)
cata=Conv1D(word_size,1, activation='relu')(cata)
cata=Conv1D(word_size,1, activation='relu')(cata)
cata=Conv1D(word_size,1, activation='relu')(cata)
cata=GlobalMaxPooling1D()(cata)
cata=Dense(1,activation='sigmoid',name="catam")(cata)
cata_model=Model(t_in,cata)

def mean_squared_errorT(y_true, y_pred):
            if not K.is_tensor(y_pred):
                y_pred = K.constant(y_pred)
            y_true = K.cast(y_true, y_pred.dtype)
            return K.sqrt(K.sum(K.square(y_pred-y_true),axis=-1))
t1m=Lambda(lambda x : x[:,:,:word_size])(t)
t2m=Lambda(lambda x : x[:,:,word_size:])(t)
o1_loss=(n1/(n1+n2))*(mean_squared_errorT(t1m,o1m))
o2_loss=(n2/(n1+n2))*(mean_squared_errorT(t2m,o2m))
c_loss=K.binary_crossentropy(tagm,cata)
loss=(o1_loss+o2_loss)+(c_loss)

rae_model=Model([t_in,tag_in,n1_in,n2_in],[encoder,o1m,o2m,cata])
rae_model.add_loss(loss)
rae_model.compile(optimizer=Adam(1e-3))
rae_model.summary()


decoder_train_model=Model([t_in,n1_in,n2_in],[o1m,o2m])
decoder_train_model.add_loss(o1_loss+o2_loss)
decoder_train_model.compile(optimizer=Adam(1e-3))

evaluator=Evaluate()
print(np.shape(datae),np.shape(tage))
ep=150
now=0
while now<ep and len(train_D)!=0 :
      if os.path.exists("rae_model.weights"):
          print("loading........")
          rae_model.load_weights('rae_model.weights')
      thistory=rae_model.fit_generator(train_D.__iter__(),
                                      steps_per_epoch=len(train_D),
                                      epochs=1,
                                      callbacks=[evaluator]
                                      )
      hi=np.mean(thistory.history["loss"])
      print("loss:",hi)
      print("save....")
      rae_model.save_weights('rae_model.weights')
      print(now)
      now=now+1
      evaluator.updata_increa(0.04)
      if hi<0.01:
          break

decoder_train_model.save_weights('decoder_train_model.weights')
decoder_model.save_weights("decoder_model.weights")
encoder_model.save_weights("encoder_model.weights")
cata_model.save_weights("cata_model.weights")


"""
def np_errorT(y_truee, y_prede,N1e,N2e):
    y_truee=np.array(y_truee)
    y_prede=np.array(y_prede)
    y_truee1=y_truee[:,:,:word_size]
    y_truee2=y_truee[:,:,word_size:]
    y_prede1=y_prede[0].reshape(1,1,word_size)
    y_prede2=y_prede[1].reshape(1,1,word_size)
    return np.sqrt(np.sum(np.square(y_prede1-y_truee1),axis=-1))*N1e/(N1e+N2e)+np.sqrt(np.sum(np.square(y_prede2-y_truee2),axis=-1))*N2e/(N1e+N2e)

print(np.shape(datae),np.shape(Ne),np.shape(tage))
for i1,i2 in enumerate(datae):
            print(str(i1)+"/"+str(len(datae)))
            num=0
            ind=i2
            ind=np.array(ind).reshape(len(ind),1,word_size)
            while len(ind)>1:
                if num>=len(ind)-1:
                    print("updata:",np.shape(ind))
                    num=0
                trb=np.concatenate([ind[num],ind[num+1]],-1).reshape(1,1,word_size*2)
                Ne[i1][num+1].reshape(1,1)
                cc=decoder_train_model.train_on_batch([trb,Ne[i1][num].reshape(1,1),Ne[i1][num+1].reshape(1,1)],[])
                pretrb=decoder_model.predict(trb)
                cha=np_errorT(trb,pretrb,Ne[i1][num],Ne[i1][num+1])
                if cha<0.2:
                    ind[num]=encoder_model.predict(trb)
                    ind=np.delete(ind,num+1,axis=0)
                    Ne[i1][num]=Ne[i1][num]+Ne[i1][num+1]
                    Ne[i1]=np.delete(Ne[i1],num+1,axis=0)
                    num=num+1
                
            if len(ind)==1:
                print("over 1")
                evaluator.result["data"].append(ind)
                evaluator.result["tag"].append(tage[i1])
                

"""


def fen_model():
        fen_in=Input(shape=(None,word_size))
        fen=fen_in
        fen=Conv1D(word_size,1, activation='relu')(fen)
        fen=Conv1D(word_size,1, activation='relu')(fen)
        fen=Conv1D(word_size,1, activation='relu')(fen)
        fen=Conv1D(word_size,1, activation='relu')(fen)
        fen=Conv1D(word_size,1, activation='relu')(fen)
        fen=Conv1D(word_size,1, activation='relu')(fen)
        fen=GlobalMaxPooling1D()(fen)
        fen_end=Dense(1,activation='sigmoid',name="fenm")(fen)
        fen_model=Model(fen_in,fen_end)
        fen_model.compile(optimizer=Adam(1e-3),loss='binary_crossentropy',metrics=['accuracy'])
        fen_model.summary()
        return fen_model
    

"""
filelist=[]
fend1=3
fend2=2

filelist.append("BioASQ-trainingDataset4b.json")

train_F=gen_data(filelist,"f")
dataef=copy.deepcopy(train_F.data)
Nef=copy.deepcopy(train_F.N)
tagef=copy.deepcopy(train_F.tag)
word_size=train_F.word_size
evaluatorf=Evaluate()

ep=150
now=0
while now<ep :
      if now!=0:
          decoder_train_model.load_weights('decoder_train_model.weights')
      thistory=decoder_train_model.fit_generator(train_F.iter__(),
                                      steps_per_epoch=len(train_F),
                                      epochs=1,
                                      callbacks=[evaluatorf]
                                      )
      decoder_train_model.save_weights('decoder_train_model.weights')
      print(now)
      now=now+1
"""

print(np.shape(evaluator.result["data"]),np.shape(evaluator.result["tag"]))
fenlei_model=fen_model()


while True:
    if os.path.exists("fenlei_model.weights"):
            fenlei_model.load_weights("fenlei_model.weights")
    hit=fenlei_model.fit(np.array(evaluator.result["data"]).reshape(len(evaluator.result["data"]),1,word_size),np.array(evaluator.result["tag"]).reshape(len(evaluator.result["tag"]),1),batch_size=64,epochs=1)
    print(hit.history)
    fenlei_model.save_weights("fenlei_model.weights")