
from sklearn import svm
from keras.datasets import mnist
import numpy as np
import sklearn.externals.joblib as joblib 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
datavol=1000
x_train=x_train[:datavol]
y_train=y_train[:datavol]
x_test=x_test[:int(datavol/10)]
y_test=y_test[:int(datavol/10)]
tag=[0,0,0,0,0,0,0,0,0,0]
tagdivi=[[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
alldata=[[[],[],[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[],[],[]]]
for ind,i in enumerate((y_train,y_test)):
    for indx,i1 in enumerate(i):
        tag[i1]+=1
        tagdivi[ind][i1]+=1
        alldata[ind][i1].append((x_train,x_test)[ind][indx])
print(tag)
print(tagdivi)

guo=[[[0,8,6,3,5],[0,8,6],[0,8],[3],[0],[4,9],[4],[2,7],[2]],[[4,9,2,7,1],[3,5],[6],[5],[8],[2,7,1],[9],[1],[7]]]

result=[]
for i in range(9):
    svc = svm.LinearSVC()
    train_data=[]
    tag_data=[]
    for i1 in guo[0][i]:
        train_data+=alldata[0][i1]
        tag_data+=list(np.zeros(shape=(len(alldata[0][i1]),1),dtype='int32'))
    print(len(train_data),guo[0][i])
    for i1 in guo[1][i]:
        train_data+=alldata[0][i1]
        tag_data+=list(np.ones(shape=(len(alldata[0][i1]),1),dtype='int32'))
    print(len(train_data),guo[1][i])
    train_data=np.array(train_data).reshape(len(train_data),28*28)
    tag_data=np.array(tag_data)
    print(np.shape(train_data),np.shape(tag_data))
    svc.fit(train_data,tag_data)
    joblib.dump(svc,str(i)+'.m')
    result.append(str(i)+'.m')

TP=[0,0,0,0,0,0,0,0,0,0]
FP=[0,0,0,0,0,0,0,0,0,0]
FN=[0,0,0,0,0,0,0,0,0,0]
cho=[[1,5],[2,3],[4],[],[],[6,7],[],[8]]
for ind,i in enumerate(x_test):
    test=np.reshape(i,(1,28*28))
    indx=0
    while True:
        tag=svc = joblib.load(result[indx]).predict(test)[-1]
        print(tag)
        if len(guo[tag][indx])==1:
            pre_tag=guo[tag][indx][-1]
            print(pre_tag,y_test[ind])
            if pre_tag==y_test[ind]:
                TP[pre_tag]+=1
            else:
                FN[pre_tag]+=1
                FP[y_test[ind]]+=1
            break
        indx=cho[indx][tag]
F1=[]
P=[]
R=[]
mf=0
mp=0
print("TP:",TP)
print("FP:",FP)
print("FN:",FN)
for i in range(10):
    P.append(TP[i]/(TP[i]+FP[i]))
    R.append(TP[i]/(TP[i]+FN[i]))
    F1.append(2*TP[i]/(2*TP[i]+FN[i]+FP[i]))
    mf+=F1[-1]
    mp+=P[-1]
print("Precision:",P)
print("Recall:",R)
print("F1:",F1)
print("mf:",mf/10)
print("mp:",mp/10)