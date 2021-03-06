
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


from keras.callbacks import Callback
import numpy as np
import keras.backend as K
from keras.layers import *
from keras.models import Model
class Evaluate(Callback):
    def __init__(self):
        self.chulin=0.01
        self.best = 0.
        self.passed = 0
        self.stage = 0
        self.result={}
        self.result["data"]=[]
        self.result["tag"]=[]
    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，不warmup有不收敛的可能。
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * 1e-3
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
    def on_epoch_end(self, epoch, logs=None):
        EMAer.apply_ema_weights()
        #data N decoder_model encoder_model tag
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
            while i3+1<len(i2):
                indata=np.concatenate([i2[i3],i2[i3+1]],-1)
                o1=debug_model.predict(indata)
                print(o1)
    def ju_juli(t1,t2,o1,o2,n1,n2,tag):# 0相似 1不相似
        def mean_absolute_error(y_true, y_pred):
            y_true=np.array(y_true)
            y_pred=np.array(y_pred)
            return np.mean(np.abs(y_pred - y_true), axis=-1)
        o1_loss=(n1/(n1+n2))*(mean_squared_error(t1,o1))
        o2_loss=(n2/(n1+n2))*(mean_squared_error(t2,o2))
        if o1_loss+o2_loss>0.2:
            return False
        else:
            if cata_model.predict(np.concatenate([t1,t2],-1))==tag:
                return True
            else:
                return False

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
for i in range(1,6):
    for i1 in range(1,6):
        if i ==1:
            if i1==4:
                break
        filelist.append("phaseB_"+str(i)+"b_0"+str(i1)+".json")
train_D=gen_data(filelist)
data=train_D.data
N=train_D.N
tag=train_D.tag
word_size=train_D.word_size


t_in=Input(shape=(None,word_size*2))
tag_in=Input(shape=(None,))
n1_in=Input(shape=(None,))
n2_in=Input(shape=(None,))
t,tag,n1,n2=t_in,tag_in,n1_in,n2_in
t=Dense(word_size*2)(t)

debug_model=Model(t_in,t)
"""
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
o1=Lambda(lambda x : x[:,:,:word_size])(dc)
o2=Lambda(lambda x : x[:,:,word_size:])(dc)
decoder_model=Model([t_in],[o1,o2])

cata=dilated_gated_conv1d(seq=encoder, dilation_rate=1)
cata=Lambda(lambda x : K.sum(x,axis=0))(cata)
cata=Dense(2,activation='softmax')(cata)
cata_model=Model([t_in,tag_in],cata)

def mean_squared_error(y_true, y_pred):
            if not K.is_tensor(y_pred):
                y_pred = K.constant(y_pred)
            y_true = K.cast(y_true, y_pred.dtype)
            return K.mean(K.square(y_pred - y_true), axis=-1)
t1=Lambda(lambda x : x[:,:,:word_size])(t)
t2=Lambda(lambda x : x[:,:,word_size:])(t)
o1_loss=(n1/(n1+n2))*(mean_squared_error(t1,o1))
o2_loss=(n2/(n1+n2))*(mean_squared_error(t2,o2))
c_loss=K.categorical_crossentropy(tag,cata)
loss=(o1_loss+o2_loss)+(c_loss)

rae_model=Model([t_in,tag_in,n1_in,n2_in],[o1,o2,cata])
rae_model.add_loss(loss)
rae_model.compile(optimizer=Adam(1e-3))
rae_model.summary()

EMAer = ExponentialMovingAverage(rae_model)
EMAer.inject()

evaluator=Evaluate()
"""
indata=np.concatenate([data[1][1],data[1][2]],-1)
indata=np.array(indata).reshape(1,1,word_size*2)
o=debug_model.predict(indata)
print("输出",np.array(o).shape)
print("输入",indata.shape)