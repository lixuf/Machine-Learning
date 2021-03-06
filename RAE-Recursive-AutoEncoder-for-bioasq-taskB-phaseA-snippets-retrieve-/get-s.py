import requests
from lxml import etree
from keras.preprocessing.text import text_to_word_sequence as sq2wsq
import json
from requests.adapters import HTTPAdapter
import time

word_size=150
header={"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134"}
file=[]
file.append("phaseA_4b_01.json")

for f_name in file:
    print(f_name)
    f=open(f_name,encoding="utf-8")
    phase=json.load(f)
    phase=phase["questions"]
    for e1,e2,in enumerate(phase):
        body=e2["body"]
        body=sq2wsq(body)
        mid=[]
        for i in body:
            if len(i)>2 and i not in stopword:
                mid.append(i)
        body=mid
        xm='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id='+UID+'&retmode=abstract&rettype=text'
        try:
            r=requests.get(xm,headers=header,timeout=15)
        except:
            r=requests.get(xm,headers=header,timeout=15)