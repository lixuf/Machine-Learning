total_data={} #全部数据存储 text aspect
model=4 #划分数据集与训练集
order=['from','to','polarity','term']#aspect格式 

#读取xml并存储 索引保持str
import  xml.dom.minidom
dom = xml.dom.minidom.parse('Laptop_Train_v2.xml')
root = dom.documentElement
all_data = root.getElementsByTagName('sentence')
counter=0
for i in all_data:
    id=i.getAttribute("id")
    counter+=1
    data={}
    text=i.getElementsByTagName('text')
    for i1 in text:
        data['text']=i1.firstChild.data
    as_data=[]
    aspectterm=i.getElementsByTagName('aspectTerms')
    for i2 in aspectterm:
        aspect=i2.getElementsByTagName('aspectTerm')
        ass_data=[]
        for i1 in aspect:
            ass_data.append(i1.getAttribute("from"))
            ass_data.append(i1.getAttribute("to"))
            ass_data.append(i1.getAttribute("polarity"))
            ass_data.append(i1.getAttribute("term"))
        as_data.append(ass_data)
    data['aspect']=as_data
    total_data[id]=data
print('counter',counter)
print('len(total_data)',len(total_data))
print(total_data["76"])