import requests
from lxml import etree
from keras.preprocessing.text import text_to_word_sequence as sq2wsq
import json
from requests.adapters import HTTPAdapter
import time
header={"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134"}
header2={"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Chorm/17.17134"}
filelist=[]

filelist.append("BioASQ-trainingDataset4b.json")


for f_name in filelist:
    print(f_name)
    f=open(f_name,encoding='utf-8')
    phase=json.load(f)
    phase1=phase["questions"]
    for n1,i in enumerate(phase1):
        print(str(n1)+"/"+str(len(phase1)))
        if n1==4:#打断处
            break
        question=i["body"]
        if "snippets" not in i:
                print(i)
                print(f_name)
        else:
                chang=len(phase1[n1]["snippets"])
                for n2,i1 in enumerate(phase1[n1]["snippets"]):
                    print(n1,str(n2)+"/"+str(chang))
                    if n2==chang:
                        break
                    doc=phase1[n1]["snippets"][n2]["document"]
                    text=phase1[n1]["snippets"][n2]["text"]
                    time.sleep(30)
                    try:
                        r = requests.get(doc,headers=header,timeout=10)
                    except :
                        print("error")
                        time.sleep(60)
                        try:
                            r = requests.get(doc,headers=header,timeout=10)
                        except:
                            print("error2")
                            time.sleep(120)
                            try:
                                r=requests.get(doc,headers=header,timeout=10)
                            except:
                                print("error3")
                                time.sleep(180)
                                try:
                                    r=requests.get(doc,headers=header,timeout=15)
                                except:
                                    print("error4")
                                    time.sleep(360)
                                    r=requests.getattr(doc,header=header2,timeout=20)
                    r_text=r.content.decode("utf-8")
                    r_text=bytes(bytearray(r_text, encoding='utf-8'))
                    html=etree.HTML(r_text)
                    html_data=html.xpath('//*[@id="maincontent"]/div/div[5]/div/div[4]/div/p[1]/text()')
                    for ine,con in enumerate(html_data):
                        sta=con.find(text)
                        if sta!=-1:
                           
                            html_data[ine]=con[:sta]+con[sta+len(text):]
                        html_data[ine]=sq2wsq(html_data[ine],";",split=".")
                        tii=0
                        for n3,juzi in enumerate(html_data[ine]):
                            if len(juzi)<12:
                                continue
                            if tii==0:
                                tii=tii+1
                                phase["questions"][n1]["snippets"][n2]["text"]=juzi
                                
                            else:
                                mid={}
                                mid["text"]=juzi
                                phase["questions"][n1]["snippets"].append(mid)
                                

    json.dump(
                phase,
                open('../PythonApplication15/'+f_name+"F.json", 'w'),
                indent=4
            )