from sklearn.tree import DecisionTreeClassifier as dtc
import numpy as np
import matplotlib.pyplot as plt

file=[["静止.txt","坐立.txt"],["桑静态.txt","壮壮静态.txt"],["站立.txt"],["坐立.txt"]]
data=[]
tag=[]
for ind,i in enumerate(file):
    for i2 in i:
        flag=0
        data_one=[]
        with open(i2,'r') as fins:
            for line in fins:
                line=line.strip('\n').split(" ")
                if len(line)<2:
                    print(line)
                    continue
                data_one.append(float(line[2]))
                data_one.append(float(line[5]))
                data_one.append(float(line[8]))
                flag+=1
                if flag>1:
                    flag=0
                    data.append(data_one)
                    data_one=[]
                if flag==1:
                    tag.append(ind)
data=np.array(data)
tag=np.array(tag)
print(np.shape(data),np.shape(tag))


tree= dtc()
tree.fit(data,tag)


pr=tree.predict([[-0.0322265625,1.025390625,0.02294921875,1.3419455680022074,-0.019654128844784777,-0.7216420869009513]])
print(pr)
#静坐az = -0.0322265625 ay = 1.025390625 az = 0.02294921875 roll = 1.3419455680022074 pitch = -0.019654128844784777 yaw = -0.7216420869009513