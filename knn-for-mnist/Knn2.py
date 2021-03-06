


import numpy as np
from keras.datasets import mnist
import os
(x_train, y_train), (x_test, y_test) = mnist.load_data()
cha=list(np.ones(shape=(10,)))
for i in y_train:
    mid=int(i)
    cha[i]+=1
print(cha)
test=input("测试数据数量：")
test=int(test)
batch=128
min=20#缩小空间
k_list=[]
k_tag=[]
k=50

for i1 in range(test):
        os.system("cls")
        print("第"+str(i1+1)+"批：%"+str(i1/test))
        for i in range(k):#初始化k池
            k_list.append(min+1)
            k_tag.append(-1)
        xtest=x_test[i1]
        xtest=np.expand_dims(xtest,axis=0)
        ytest=y_test[i1]
        _mid=xtest.copy()
        tt=0
        for i2 in range(batch-1):
            xtest=np.concatenate([xtest,_mid],axis=0)
        for i3 in range(int(len(x_train)/batch)):
            print(str(i3)+"/"+str(int(len(x_train)/batch)))
            xtrain=x_train[i3*batch:(i3+1)*batch]
            ytrain=y_train[i3*batch:(i3+1)*batch]
            _post=np.mean(np.sqrt(np.sum(np.square(xtest-xtrain),axis=-1)),axis=1)
            _stu=np.where(_post<min,True,False)
            for i4,ex in enumerate(_stu):
                if ex:
                    tt+=1
                    for i5 in range(k):#更新k池 采用边搜索边排序的方法优化排序 每次找到最小的都去替换k池中第一个比它大的
                        if k_list[i5]>_post[i4]:
                            k_list[i5]=_post[i4]
                            k_tag[i5]=ytrain[i4]
                            break
        print(k_list)
        print(k_tag)
        print(tt)
        vote=list(np.zeros(shape=(10,)))
        max_num=0
        max_tag=-1
        for i6 in range(k):
            _tag=k_tag[i6]
            _tag=int(_tag)
            if _tag<0:
                break
            vote[_tag]+=1
            if max_num< vote[_tag]:
                max_num=vote[_tag]
                max_tag=_tag
        print(vote)
        print("完成\n预测值:",str(max_tag))
        print("真实值",ytest)
        en=input("是否继续")
