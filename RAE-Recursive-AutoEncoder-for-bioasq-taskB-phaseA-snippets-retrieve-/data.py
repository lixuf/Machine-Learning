import json
from keras.preprocessing.text import text_to_word_sequence as sq2wsq
from nltk.tokenize import word_tokenize,sent_tokenize
import numpy as np
import os
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
import multiprocessing
import copy
class gen_data():
    def __init__(self,f_name,swi="r",gen="r",**awig):
        self.word2id={}
        self.id2word={}
        self.data=[]
        self.N=[]
        self.tag=[]
        self.swi=swi
        self.maxlen=32
        self.dtype_word="float32"
        self.word_size=150
        """
        data question+snippt已经分好词
        tag  的索引跟data对应
        """
        self.id=1
        self.exc=0
        if gen=="r":
            if  (os.path.exists("word2vec_lstm.model")):
                self.model=Word2Vec.load("word2vec_lstm.model")
                self.exc=1
                for i in f_name:
                    print("处理w2v"+i)
                    self.processdata(i)
                
            else:
                for i in f_name:
                    print("处理"+i)
                    self.processdata(i)
                self.random()
                if self.exc==0:
                    self.word2vec()
                self.model=Word2Vec.load("word2vec_lstm.model")
                self.exc=1
                for i in f_name:
                    print("处理w2v"+i)
                    self.processdata(i)
            self.random()
        else:
            self.data=awig["da"]
            self.N=awig["na"]
            self.tag=np.ones(shape=(len(self.data)))
    def processdata(self,f_name):
        swich=0
        stop=100000000000000000
        if f_name[-6]=="F":
            print("swich",swich)
            swich=1
            stop=1
        f=open(f_name,encoding='utf-8')
        phase=json.load(f)
        phase=phase["questions"]
        for i in phase:
            print(i["body"])
            question=word_tokenize(i["body"])
            print(question)
            a=input()
            qu1=0
            for i1 in range(len(question)):
                if question[i1-qu1] not in self.word2id:
                    self.word2id[question[i1-qu1]]=self.id
                    self.id2word[self.id]=question[i1-qu1]
                    self.id=self.id+1
                if self.exc==1:
                    if len(question[i1-qu1])==1:
                        print("去掉",question[i1-qu1])
                        del question[i1-qu1]
                        qu1=qu1+1
                    else:
                        if question[i1-qu1] not in self.model.wv:
                            print(question[i1-qu1]+"缺失")
                            del question[i1-qu1]
                            qu1=qu1+1
                        else:
                            question[i1-qu1]=np.array(self.model.wv[question[i1-qu1]]).reshape(1,self.word_size)
            if "snippets" not in i:
                print(i)
                print(f_name)
            else:
               for kk,i1 in enumerate(i["snippets"]):
                    if swich==1 and stop!=0:
                       if stop==kk:
                           break
                    print(i1["text"])
                    text=word_tokenize(i1["text"])
                    print(text)
                    b=input()
                    qu2=0
                    for i2 in range(len(text)):
                        if len(text[i2-qu2])==1:
                            print("去掉",text[i2-qu2])
                            del text[i2-qu2]
                            qu2=qu2+1
                        else:
                            if text[i2-qu2] not in self.word2id:
                                self.word2id[text[i2-qu2]]=self.id
                                self.id2word[self.id]=text[i2-qu2]
                                self.id=self.id+1
                            if self.exc==1:
                                if text[i2-qu2] not in self.model.wv:
                                    print(text[i2-qu2]+"缺失")
                                    del text[i2-qu2]
                                    qu2=qu2+1
                                else:
                                    text[i2-qu2]=np.array(self.model.wv[text[i2-qu2]]).reshape(1,self.word_size)
                    
                    if swich==0:
                        self.tag.append(1)
                        self.N.append(np.ones(shape=(len(question+text),)))
                        self.data.append(question+text)
                    else:#qa是1 不是qa是0
                        self.tag.append(0)
                        self.N.append(np.ones(shape=(len(text),)))
                        self.data.append(text)
  
                    if len(self.data[-1])!=len(self.N[-1]):
                            print("长度不匹配",len(self.data[-1]),len(self.N[-1]))
    def word2vec(self):#用词做索引
            path = get_tmpfile("word2vec_lstm.model")
            model = Word2Vec(self.data, size=self.word_size, window=2, min_count=1,
                             workers=multiprocessing.cpu_count())
            model.save("word2vec_lstm.model")
    def __iter__(self):
        N1=[]
        N2=[]
        T1=[]
        T2=[]
        Tag=[]
        T=[]
        print("gen start..",self.maxlen)
        while True:
                for i1,i2 in enumerate(self.data):
                        for i3,i4 in enumerate(i2):
                            if i3+1>=len(i2):
                                continue
                            T1.append(self.data[i1][i3])
                            T2.append(self.data[i1][i3+1])
                            N1.append(self.N[i1][i3])
                            N2.append(self.N[i1][i3+1])
                            Tag.append(self.tag[i1])
                            if len(T1)==self.maxlen:
                                T1=np.array(T1)
                                T2=np.array(T2)
                                Tag=np.array(Tag)
                                N1=np.array(N1)
                                N2=np.array(N2)
                                T=np.concatenate([T1,T2],-1).reshape(self.maxlen,1,self.word_size*2)
                                if self.swi=="r":
                                    yield [T, Tag,N1,N2], None
                                else :
                                    yield [T,N1,N2], None
                                T,T1,T2, Tag, N1,N2 = [],[], [], [], [],[]# Tag是qa关系标签 N是孩子数量 起始为1
                
 
    def random(self):
        random_order =list(range(len(self.data)))
        np.random.shuffle(random_order)
        datacopy = [self.data[j] for i, j in enumerate(random_order)]
        Ncopy=[self.N[j] for i, j in enumerate(random_order)]
        tagcopy=[self.tag[j] for i, j in enumerate(random_order)]
        self.data=datacopy
        self.N=Ncopy
        self.tag=tagcopy
    def __len__(self):
        step=0
        print("检查数据：")
        print("len data",len(self.data))
        print("len tag",len(self.tag))
        print("len N",len(self.N))
        for i in self.data:
                step=step+len(i)-1
        if step<self.maxlen:
            self.maxlen=step
        temp=0.0
        if self.maxlen==0:
            return 0
        else:
            temp=step/self.maxlen
            if temp-int(temp)>0:
                temp=int(temp)+1
            return  int(temp)
    def updata(self,shujud,shujun,**jwig):
        self.data=copy.deepcopy(shujud)
        self.N=copy.deepcopy(shujun)
        if "shujutag" in jwig:
            self.tag=copy.deepcopy(jwig["shujutag"])
        else:
            self.tag=np.ones(shape=(len(self.data)))
        print("updata::len data",len(self.data))
        print("len train_",self.__len__())

"""
filelist=[]
for i in range(1,6):
    for i1 in range(1,6):
        if i ==1:
            if i1==4:
                break
        filelist.append("phaseB_"+str(i)+"b_0"+str(i1)+".json")
c=gen_data(filelist)

"""
"""
                                while i<len(self.data):
                    if mid!=0:
                        i1=mid
                        mid=0
                    else:
                        i1=0
                    while i1 <len(self.data[i]):
                        if i1+1!=len(self.data[i]):
                            if counter>=self.maxlen:
                                counter=0
                                i=i-1
                                mid=i1
                                break
                            T1.append(self.data[i][i1])
                            T2.append(self.data[i][i1+1])
                            N1.append(self.N[i][i1])
                            N2.append(self.N[i][i1+1])
                            Tag.append(self.tag[i])
                            counter=counter+1
                        i1=i1+1
                    if len(T1)==self.maxlen:
                        T1=np.array(T1)
                        T2=np.array(T2)
                        Tag=np.array(Tag)
                        N1=np.array(N1)
                        N2=np.array(N2)
                        T=np.concatenate([T1,T2],-1)
                        
                        if self.swi=="r":
                            yield [T, Tag,N1,N2], None
                        else :
                            yield [T,N1,N2], None
                        T,T1,T2, Tag, N1,N2 = [],[], [], [], [],[]# Tag是qa关系标签 N是孩子数量 起始为1
                    else:
                        print("error!!!!!")
                    i=i+1
"""