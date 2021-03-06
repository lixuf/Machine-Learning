import numpy as np
class mut_linear():
    def __init__(self,ins):
        self.x=ins[0]
        self.y=ins[1]
        self.call()    
    def _jiayi(self):
        return np.concatenate([self.x,np.ones(shape=(len(self.x),1))],axis=-1)
    def call(self):
        ins=np.mat(self._jiayi())
        self.wb=(ins.T*ins).I*ins.T*np.mat(self.y)
    def printf(self):
        print("wb:",self.wb)
    def predict(self,ins):
        return np.mat(np.concatenate([ins,[[1.0]]],axis=0)).T*self.wb

ins=[np.array([[1,6,11],[2,8,8.5],[3,10,15],[4,14,18],[5,18,11]]),np.array([[9.7759],[10.7522],[12.7048],[17.5863],[13.6811]])]
learn=mut_linear(ins)
learn.printf()
res=np.array([[2],[10],[11]])
res=learn.predict(res)
print("price",res)