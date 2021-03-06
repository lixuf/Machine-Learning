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
                o1,o2=decoder_model.predict(np.concatenate([i2[i3],i2[i3+1]],-1))
                if self.ju_juli(i2[i3],i2[i3+1],o1,o2,N[i1][i3],N[i1][i3+1],tag[i1]):
                    datanew.append(encoder_model.predict(np.concatenate([i2[i3],i2[i3+1]],-1)))
                    Nnew.append(N[i1][i3]+N[i1][i3+1])
                    i3=i3+1
                else:
                    datanew.append(i2[i3])
                    Nnew.append(N[i1][i3])
                    if i3+2==len(i2):
                        datanew.append(i2[i3+1])
                        Nnew.append(N[i1][i3+1])
                i3=i3+1
            data[i1]=datanew
            N[i1]=Nnew
        for i in erase:
            del data[i]
            del N[i]
        if len(erase)/len(data)>self.chulin:
            self.chulin=len(erase)/len(data)
            print("处理个数",self.chulin)
            rae_model.save_weights('best_model.weights')
        EMAer.reset_old_weights()
        if epoch + 1 == 50 or (
            self.stage == 0 and epoch > 10 
        ):
            self.stage = 1
            rae_model.load_weights('best_model.weights')
            EMAer.initialize()
            K.set_value(self.model.optimizer.lr, 1e-4)
            K.set_value(self.model.optimizer.iterations, 0)
            opt_weights = K.batch_get_value(self.model.optimizer.weights)
            opt_weights = [w * 0. for w in opt_weights]
            K.batch_set_value(zip(self.model.optimizer.weights, opt_weights))
  
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