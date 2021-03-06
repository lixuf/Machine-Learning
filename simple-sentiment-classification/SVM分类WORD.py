import numpy as np
import gensim
import jieba
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
from sklearn import svm
from sklearn.preprocessing import Imputer
from sklearn.metrics import classification_report

def divide_data(name):
    p_data=open(name+'p.txt','w',encoding='GB18030')
    o_data=open(name+'o.txt','w',encoding='GB18030')
    with open(name+'.txt','r',encoding='GB18030') as file:
        for line in file:
            if line[0]=='1':
                p_data.write(line)
            else:
                o_data.write(line)

    p_data.close()
    o_data.close()
def cleandata(name):
   
    def predo(name):  
        file_out = open(name+'2.txt', 'w', encoding='utf-8')
        with open(name+'.txt', 'r', encoding="GB18030") as file_object:
            for line in file_object:
                line = line.strip()
                file_out.write(line+"\n")
        file_out.close()
    def jie(name):
    
        stop_words_file = "stop_words.txt"
        stop_words = list()
        with open(stop_words_file, 'r', encoding="GBK") as stop_words_file_object:
            contents = stop_words_file_object.readlines()
            for line in contents:
                line = line.strip()
                stop_words.append(line)

        origin_file = name+"2.txt"
        target_file = open(name+"3.txt", 'w', encoding="utf-8")
        with open(origin_file, 'r', encoding="utf-8") as origin_file_object:
            contents = origin_file_object.readlines()
            for line in contents:
                line = line.strip()
                out_str = ''
                word_list = jieba.cut(line, cut_all=False)
                for word in word_list:
                    if word not in stop_words:
                        if word != "\t":
                            out_str += word
                            out_str += ' '
                target_file.write(out_str.rstrip()+"\n")

        target_file.close()
        print("end")
    predo(name)
    jie(name)
def getvector(name,yorn):
    model = gensim.models.Word2Vec.load("word2vec.model")
    if yorn == '1':
        re_train_model(name)
    with open(name+'.txt','r',encoding='utf-8') as indata:
        
        x_data=indata.readlines()
   
    vecs= getVecs(model, x_data, 400)
    print('vector putout scu')
    return vecs
def getVecs(model,indata,dimension):
    vecs=[]
    for words in indata:
        acount_continue=0
        words=liststring(words)
        vec=np.zeros(400)
        for index in words:
            try:
                vec+=model[index.strip()]
            except KeyError:
                acount_continue+=1
                continue
        vec/=(len(words)-acount_continue)
        vecs.append(vec)
    return vecs
def liststring(indata):
    listwords=[]
    words=''
    for word in indata:
        if word==' ':
            if words!=' ' and words!='':
                listwords.append(words)
            words=''
        words+=word
    return listwords
def re_train_model(name):
    model=gensim.models.Word2Vec.load("word2vec.model")
    model.train(LineSentence(name+'.txt'), total_examples=1, epochs=1)
    model.save("word2vec.model")
    print('model upload success')
def save_data(indata,name):
    np.savetxt(name+'.txt',indata)
def svm_train_test(data,tag,testdata,testtag):
    print('start train')
    clf = svm.LinearSVC()
    data = Imputer().fit_transform(data)
    clf_res = clf.fit(data,tag)  
    print('end train\nstart test')
    testdata = Imputer().fit_transform(testdata)
    test_pred = clf_res.predict(testdata)
    print(classification_report(testtag, test_pred))







#divide_data('测试')
#cleandata('训练')
#cleandata('测试p')
#cleandata('测试o')
#train_vec=getvector('训练3','')
#save_data(train_vec,'tvec')
#testp_vec=getvector('测试p3','')
#save_data(testp_vec,'epvec')
#testo_vec=getvector('测试o3','')
#save_data(testo_vec,'eovec')
#print('vevtor done')
print('loads.....')
train_vec=np.loadtxt('tvec.txt',dtype=np.float64)
testp_vec=np.loadtxt('epvec.txt',dtype=np.float64)
testo_vec=np.loadtxt('eovec.txt',dtype=np.float64)

p_tag = np.ones(31714)
n_tag = np.zeros(31060)
tag=np.hstack((n_tag,p_tag))
#test
test_p_tag=np.ones(9323)
test_n_tag=np.zeros(10431)
testdata=np.concatenate((testp_vec,testo_vec),axis = 0)
testtag=np.hstack((test_p_tag,test_n_tag))
svm_train_test(train_vec,tag,testdata,testtag)  