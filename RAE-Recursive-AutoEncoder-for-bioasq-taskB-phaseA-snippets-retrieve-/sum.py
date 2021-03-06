import requests
from lxml import etree
from keras.preprocessing.text import text_to_word_sequence as sq2wsq
import json
from requests.adapters import HTTPAdapter
import time
from data import gen_data
import numpy as np
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from nltk.tokenize import word_tokenize,sent_tokenize
from keras.callbacks import Callback
import os
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
import multiprocessing
import copy
word_size=150
model=Word2Vec.load("word2vec_lstm.model")
header={"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134"}

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


decoder_train_model.load_weights('decoder_train_model.weights')
decoder_model.load_weights("decoder_model.weights")
encoder_model.load_weights("encoder_model.weights")
cata_model.load_weights("cata_model.weights")


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
fenlei_model=fen_model()
fenlei_model.load_weights("fenlei_model.weights")








class Evaluate(Callback):
    def __init__(self,x,y,na,train_F):
        self.datae=y
        self.nae=na
        self.retu=x
        self.counter=list(range(len(self.retu)))
        self.gat=0.2
        self.train_F=train_F
    def on_epoch_end(self, epoch, logs=None):
        upd=[]
        upn=[]
        for tt,i in enumerate(self.counter):
            if len(self.datae[i])==1:
                self.retu[i]["words"]=self.datae[i]
                del self.counter[tt]
                continue
            indata=[]
            indatae=[]
            inN1=[]
            inN2=[]
            for e1 in range(len(self.datae[i])):
                if e1+1==len(self.datae[i]):
                    indatae.append(self.datae[i][e1])
                    break
                indatae.append(self.datae[i][e1])
                indata.append(np.concatenate([self.datae[i][e1],self.datae[i][e1+1]],axis=-1).reshape(1,word_size*2))
                inN1.append(self.nae[i][e1])
                inN2.append(self.nae[i][e1+1])
            indata=np.array(indata)
            inN1=np.array(inN1)
            inN2=np.array(inN2)
            o1,o2=decoder_model.predict(indata)
            encoderdata=[]
            encoderdata=encoder_model.predict(indata)
            datanew=[]
            Nnew=[]
            tiao=0
            for i3,ex in enumerate(self.ju_juli(indata,o1,o2,inN1,inN2,i)):
                if tiao==1:
                    tiao=0
                    continue
                if ex:
                    datanew.append(encoderdata[i3].reshape(word_size))
                    Nnew.append(inN1[i3]+inN2[i3])
                    tiao=1
                else:
                    datanew.append(np.array(indatae[i3]).reshape(word_size))
                    Nnew.append(inN1[i3])
                    if i3+2==len(indatae):
                        datanew.append(np.array(indatae[i3+1]).reshape(word_size))
                        Nnew.append(inN2[i3])
            self.datae[i]=np.array(datanew).reshape(len(datanew),word_size)
            self.nae[i]=Nnew
            if len(self.datae[i])==1:
                self.retu[i]["words"]=self.datae[i]
                del self.counter[tt]
            else:
                upd.append(datanew)
                upn.append(Nnew)
        if len(self.counter)==0:
            self.train_F.updata([],[])
        else:
            self.train_F.updata(upd,upn)
    def ju_juli(self,indata,o1,o2,n1,n2,i):# 0相似 1不相似
        def np_error(y_truee, y_prede):
            y_truee=np.array(y_truee)
            y_prede=np.array(y_prede)
            return np.sqrt(np.sum(np.square(y_prede-y_truee),axis=-1))
        o1_losse=(n1/(n1+n2))*(np_error(indata[:,:,:word_size],o1))
        o2_losse=(n2/(n1+n2))*(np_error(indata[:,:,word_size:],o2))
        o=o1_losse+o2_losse
        o=o[:,:1]
        retu1=np.where(o<self.gat,True,False)       

    def ret(self):
        for nu,i in enumerate(self.retu):
            if "words" not in i:
                print("有错误，有words不在retu中：",nu)
                while len(self.datae[nu])!=1:
                    self.datae[nu][0]=encoder_model.predict(np.concatenate([self.datae[nu][0],self.datae[nu][1]],axis=-1).reshape(1,1,word_size*2))
                    print(len(self.datae[nu]))
                    self.datae[nu]=np.delete(self.datae[nu],1,axis=0)
                self.retu[nu]["words"]=self.datae[nu]
        print(len(self.retu))
        return self.retu
    def updata_increa(self,pp):
        self.gat=self.gat+pp





def w2v(x):
    re=np.zeros(shape=(len(x),word_size))
    erase=[]
    for i,co in enumerate(x):
        if co in model.wv:
            re[i]=model.wv[co]
        else:
            erase.append(i)
            print(co+"不存在")
    e=0
    for i in erase:
        re=np.delete(re,i-e,axis=0)
        e=e+1
    return re
file=[]
file .append("phaseB_4b_01.json")

def fen(x,uid):
    y=sent_tokenize(x)
    begin=0
    end=0
    re=[]
    for i1,i2 in enumerate(y):
        begin=end
        end=begin+len(i2)+1
        temp={}
        temp["off"]=(begin,end)
        i2e=word_tokenize(i2)
        temp["words"]=w2v(i2e)
        temp["UID"]=uid
        re.append(temp)
    return re

def cmp(x):
    return x["words"]


def train():
    x=[]
    y=[]
    na=[]
    for i in test:
        temp={}
        temp["off"]=i["off"]
        temp["UID"]=i["UID"]
        x.append(temp)
        y.append(np.concatenate([question,np.array(i["words"])],axis=0))
        na.append(np.ones(shape=(len(question)+len(i["words"]))))
    train_F=gen_data("None",'f','f',na=na,da=y)
    evaluatorf=Evaluate(x,y,na,train_F)
    ep=150
    now=0
    increa=0.004
    while now<ep and len(train_F)!=0 :
          if os.path.exists("decoder_train_model.weights"):
              print("loading........")
              decoder_train_model.load_weights('decoder_train_model.weights')
          thistory=decoder_train_model.fit_generator(train_F.__iter__(),
                                          steps_per_epoch=len(train_F),
                                          epochs=1,
                                          callbacks=[evaluatorf]
                                          )
          hi=np.mean(thistory.history["loss"])
          print("loss:",hi)
          print("save....")
          decoder_train_model.save_weights('decoder_train_model.weights')
          print(now)
          evaluatorf.updata_increa(increa)
          now=now+1
          if hi<0.01:
              break
    x=evaluatorf.ret()
    for t1 in range(len(x)):
        x[t1]["words"]=(fenlei_model.predict(np.array(x[t1]["words"]).reshape(1,1,word_size)))[-1][-1]
    return x

for f_name in file:
    print(f_name)
    f=open(f_name,encoding="utf-8")
    f_t=open(f_name-".json"+"D.json",encoding="utf-8")
    phase=json.load(f)
    phase_t=json.load(f_t)
    phase=phase["questions"]
    phase_t=phase_t["questions"]
    MAP=0
    AP=[]
    copor={}
    for n1,n2 in enumerate(phase):
        gold={}
        test=[]
        ap=0
        question_a=n2["body"]
        question=word_tokenize(question_a)
        question=w2v(question)
        test_n={}
        for y1,y2 in enumerate(phase_t):
            if question_a==y2["body"]:
                test_n=y2
                break
        for n3,n4 in enumerate(n2["snippets"]):
            doc=n4["document"]
            UID=doc.split("/")[-1]
            begin=n4["offsetInBeginSection"]
            end=n4["offsetInEndSection"]
            tempstr=n4["text"]
            if UID in gold:
                gold[UID].append((begin,end))
            else:
                temp=[]
                temp.append((begin,end))
                gold[UID]=temp
        test=[]
        for n3,n4 in enumerate(test_n["snippets"]):
                ab=n4["text"]
                UID=n4["document"]
                temp=[]
                temp=fen(ab,UID)
                test=test+temp
        print("处理中...")
        test=train()#words 中应该为float
        test.sort(key=cmp,reverse=True)
        if len(test)>=10:
            test=test[:10]
        for p1 in test:
            for p2 in p1:
                print(p2,p1[p2])
        LR=0
        rec=[]
        pr=[]
        pre_pr=0
        for e1,e2 in enumerate(test):
            if e2["words"]>0.5 and e2["UID"] in gold:
                LR=LR+1
                rec.append(1)
                pre_pr=pre_pr*e1
                slen=e2["off"][1]-e2["off"][0]
                sset=set(list(range(e2["off"][0],e2["off"][1])))
                gset=[]
                for e3 in gold[e2["UID"]]:
                    gset=gset+list(range(e3[0],e3[1]))
                gset=set(gset)
                oset=gset&sset
                oset=len(oset)
                pre_pr=(oset/slen+pre_pr)/(e1+1)
                pr.append(pre_pr)
            else:
                rec.append(0)
                pr.append(0)
        for e4,e5 in enumerate(rec):
            ap=ap+pr[e4]*e5
        AP.append(ap/LR)
        print(n1,"AP",AP[-1])
    for y1 in AP:
        MAP=MAP+y1
    MAP=MAP/len(AP)
    print(f_name+" MAP:",MAP)