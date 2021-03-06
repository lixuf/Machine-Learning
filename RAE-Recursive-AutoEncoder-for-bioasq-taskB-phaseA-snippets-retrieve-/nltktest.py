import nltk
import requests
from lxml import etree
from keras.preprocessing.text import text_to_word_sequence as sq2wsq
import json
from requests.adapters import HTTPAdapter
import time
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
import re
word_size=150
header={"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134"}
file_list=[]

file_list.append("phaseA_5b_04.json")
file_list.append("phaseA_5b_05.json")
f2y={'4':2016,'5':2017,'6':2018,'7':2019}
pattern_retmax=r'<RetMax>(.*?)</RetMax>'
pattern_id=r'<Id>(.*?)</Id>'
grammar = r"""
    NP: {<DT|PRP\$>?<JJ>*<NN>}
        {<NNP>+}
"""
chunkParser = nltk.RegexpParser(grammar)


def get_np(sentence):#list 单个词
        tokens = nltk.word_tokenize(sentence)
        tokenized = nltk.pos_tag(tokens)
        chunked = chunkParser.parse(tokenized)
        re=[]
        re2=[]
        for y in tokenized:
            if y[1]=="NN" or y[1]=="NNP":
                re2.append(y[0])
        for i in chunked:
            try:
                if i.label()=="NP":
                    leaf_values = i.leaves()
                    for e in leaf_values:
                        re.append(e[0])
            except:
                print(i)
        return re,re2

def po(f_name):
    f=open(f_name,encoding="utf-8")
    file=json.load(f)
    years=f_name[-10]
    years=f2y[years]
    phase=file["questions"]
    for e1,e2,in enumerate(phase):
        sentence=e2["body"]
        npl,np2=get_np(sentence)
        term=""
        for i in npl:
            term=term+i+"+AND+"
        term=term[:-5]
        xm="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term="+term+"&mindate=1960&maxdate="+str(years)+"&datetype=edat&retmax=20&sort=relevance"
        print(xm)
        try:
            r=requests.get(xm,headers=header,timeout=15)
        except:
            r=requests.get(xm,headers=header,timeout=15)
        time.sleep(1)
        r_text=r.content.decode("utf-8")
        html_data=re.search(pattern_retmax, r_text, flags=0).group(1)
        if int(html_data)==0:
            term=""
            for i in np2:
                term=term+i+"+AND+"
            term=term[:-5]
            xm="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term="+term+"&mindate=1960&maxdate="+str(years)+"&datetype=edat&retmax=20&sort=relevance"
            try:
                r=requests.get(xm,headers=header,timeout=15)
            except:
                r=requests.get(xm,headers=header,timeout=15)
            time.sleep(0.5)
            r_text=r.content.decode("utf-8")
            html_data=re.search(pattern_retmax, r_text, flags=0).group(1)
        uid=""
        sum=0
        mid=re.findall(pattern_id, r_text)
        for t1 in range(int(html_data)):
            uid=uid+mid[t1]+','
            sum=sum+1
        uid=uid[:-1]
        print(uid)
        xm="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id="+uid+"&rettype=abstract&retmode=text"
        try:
            r=requests.get(xm,headers=header,timeout=15)
        except:
            r=requests.get(xm,headers=header,timeout=15)
        r_text=r.content.decode("utf-8")
        mid_si=[]
        for i in range(sum):
            mid_text=r_text.split("PMID: "+mid[i])
            r_text=mid_text[1]
            mid_text=mid_text[0]
            mid_text=mid_text.split("\n\n")
            mid_dict={}
            mid_n=0
            mid_len=0
            for p1,p2 in enumerate(mid_text):
                if len(p2)>mid_len:
                    mid_len=len(p2)
                    mid_n=p1
            mid_dict["document"]=mid[i]
            mid_dict["text"]=mid_text[mid_n]
            mid_si.append(mid_dict)
            """
            if i==0:
                print(mid_text[4])
                mid_dict["text"]=mid_text[4]
            else:
                print(mid_text[i[5]))
                mid_dict["text"]=mid_text[5]
            mid_si.append(mid_dict)
            """
           
        file["questions"][e1]["snippets"]=mid_si  
    file_save = open(f_name[:-5]+"D.json",'w',encoding='utf-8')
    json.dump(file,file_save,indent =4)


for f_name in file_list:
    print(f_name)
    po(f_name)