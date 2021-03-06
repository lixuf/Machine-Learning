
class Pmf():
    def __init__(self, *args, **kwargs):
        self.value={}
        self.sta={}
        self.sum={}
    def set(self,name,prob=1):
        self.value[name]=prob
        self.sta[name]={}
    def nomr(self,*indict):
        if len(indict)==0:
            indict=self.value
        else:
            indict=indict[0]
        sum=0.0
        for _,y in indict.items():
            sum+=y
        for x,_ in indict.items():
            indict[x]/=sum
    def state(self,name,inlist):
        self.nomr(inlist)
        self.sta[name].update(inlist)
    def siran(self,inlist):
        print(inlist)
        for name,_ in self.value.items():
            res1=1.0
            for i in inlist:
                if i in self.sta[name]:
                    res1*=self.sta[name][i]
            self.value[name]=res1
    def printf(self):
        self.nomr(self.value)
        for x,y in self.value.items():
            print(x,":",y)

play=Pmf()
play.set("no")
play.set("yes")
play.nomr()
name={"s":3,"r":2}
play.state("no",name)
name={"hot":2,"cool":1,"mild":2}
play.state("no",name)
name={"high":4,"nor":1}
play.state("no",name)
name={"false":2,"true":3}
play.state("no",name)
name={"s":2,"o":4,"r":3}
play.state("yes",name)
name={"hot":2,"mild":4,"cool":3}
play.state("yes",name)
name={"high":3,"nor":6}
play.state("yes",name)
name={"false":6,"true":3}
play.state("yes",name)
play.siran(["s","hot","high","true"])
play.printf()
       

        
                
