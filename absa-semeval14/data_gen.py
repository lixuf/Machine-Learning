
class data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(self.data[0].keys())
            np.random.shuffle(idxs)
            T1, T2, S1, S2 = [], [], [], []
            for i in idxs:
                text=data[0][i]["text"]
                text_un=data[1][i]["text"]
                T1.append(text_un)
                T2.append(text)
                s1, s2 = np.zeros(maxlen), np.zeros(maxlen)
                for sp in data[0][i]['aspect']:
                    s1[sp[0]] = 1
                    s2[sp[1]-1] = 1
                S1.append(s1)
                S2.append(s2) 
                if len(T1) == self.batch_size or i == idxs[-1]:
                    yield [T1, T2, S1, S2], None
                    T1, T2, S1, S2 = [], [], [], []
