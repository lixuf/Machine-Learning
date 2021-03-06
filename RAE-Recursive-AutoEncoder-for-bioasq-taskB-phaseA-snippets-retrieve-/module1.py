while True:
    T1=[]
    for i1,i2 in enumerate(self.data):
        for i3,i4 in enumerate(i2):
            T1.append(i4)
            if len(T1)==self.maxlen:
                yeild T1
                T1 = []
