total_data={} #全部数据存储 text aspect
mode=0 #划分数据集与训练集
maxlen=80#最长长度
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
    aspectterm=i.getElementsByTagName('aspectTerms')
    as_data=[]
    for i2 in aspectterm:
        aspect=i2.getElementsByTagName('aspectTerm')     
        for i1 in aspect:
            ass_data=[]
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

#repair
#1874 1650 1225
total_data['1650']['text']='With the macbook pro it comes with free security software to protect it from viruses and other intrusive things from downloads and internet surfing or emails.'
total_data['1650']["aspect"][0][3]="security software"
total_data['1225']['text']='The computer was shipped to their repair depot on june 24 and returned on July 2 seems like a short turn around time except the computer was not repaired when it was returned.'
total_data['1874']['text']='I feel that it was poorly put together, because once in a while different plastic pieces would come off of it.'
total_data['1874']["aspect"][0][3]="plastic pieces"


#process

#from to 延续 [ ， ) |分词
import keras
for i in total_data.keys():
    total_data[i]['text']=keras.preprocessing.text.text_to_word_sequence(total_data[i]['text'])
    for i1 in range(len(total_data[i]["aspect"])):
        total_data[i]['aspect'][i1][0]=total_data[i]['text'].index(keras.preprocessing.text.text_to_word_sequence(total_data[i]['aspect'][i1][3])[0])
        total_data[i]['aspect'][i1][1]=total_data[i]['text'].index(keras.preprocessing.text.text_to_word_sequence(total_data[i]['aspect'][i1][3])[-1])+1   
print(total_data["76"])


#w2v
import os
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
import multiprocessing
def word2vec(in_txt):
    path = get_tmpfile("word2vec_lstm.model")
    model = Word2Vec(in_txt, size=150, window=6, min_count=0,
                     workers=multiprocessing.cpu_count())
    model.save("word2vec_lstm.model")
if not (os.path.exists("word2vec_lstm.model")):
    print("start train word2vec......")
    text=[]
    for i in total_data.keys():
        text.append(total_data[i]["text"])
    print('len(text)',len(text))
    word2vec(text)
model=Word2Vec.load("word2vec_lstm.model")

#建立词表 id是int
import copy
id2word = {i+1:j for i,j in enumerate(model.wv.index2word)}
word2id = {j:i for i,j in id2word.items()}
total_data_un=copy.deepcopy(total_data)
for i in total_data_un.keys():
    for i2 in range(len(total_data_un[i]["text"])):
        total_data_un[i]["text"][i2]=word2id[total_data_un[i]["text"][i2]]
print(total_data_un["2334"]["text"][2])

#转化为词向量
for i in total_data.keys():
    senten=[]
    if maxlen<len(total_data[i]["text"]):
        maxlen=len(total_data[i]["text"])
    for i2 in range(0,len(total_data[i]["text"])):
        vector = model.wv[total_data[i]["text"][i2]]
        senten.append(vector)
    total_data[i]["text"]=senten
print('maxlen',maxlen)
print(total_data_un["2334"]["text"][2])

#划分数据集
import numpy as np
import json
if not os.path.exists('../random_order_vote.json'):
    random_order =list(total_data.keys())
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('../random_order_vote.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('../random_order_vote.json'))
train_data = [total_data[j] for i, j in enumerate(random_order) if i % 5!= mode]
test_data = [total_data[j] for i, j in enumerate(random_order) if i % 5 == mode]
train_data_un = [total_data_un[j] for i, j in enumerate(random_order) if i % 5!= mode]
test_data_un = [total_data_un[j] for i, j in enumerate(random_order) if i % 5 == mode]
print('len(test_data)',len(test_data))
print('len(train_data)',len(train_data))
print('len(test_data_un)',len(test_data_un))
print('len(train_data_un)',len(train_data_un))
print(len(test_data_un[6]["text"]),len(test_data[6]["text"]))
print(test_data_un[6])




#model
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return K.tf.gather_nd(seq, idxs)


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1, keepdims=True)


def dilated_gated_conv1d(seq, mask, dilation_rate=1):
    """膨胀门卷积（残差式）
    """
    dim = K.int_shape(seq)[-1]
    h = Conv1D(dim*2, 3, padding='same', dilation_rate=dilation_rate)(seq)
    def _gate(x):
        dropout_rate = 0.1
        s, h = x
        g, h = h[:, :, :dim], h[:, :, dim:]
        g = K.in_train_phase(K.dropout(g, dropout_rate), g)
        g = K.sigmoid(g)
        return g * s + (1 - g) * h
    seq = Lambda(_gate)([seq, h])
    seq = Lambda(lambda x: x[0] * x[1])([seq, mask])
    return seq


class Attention(Layer):
    """多头注意力机制
    """
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        q_in_dim = input_shape[0][-1]
        k_in_dim = input_shape[1][-1]
        v_in_dim = input_shape[2][-1]
        self.q_kernel = self.add_weight(name='q_kernel',
                                        shape=(q_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.k_kernel = self.add_weight(name='k_kernel',
                                        shape=(k_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.v_kernel = self.add_weight(name='w_kernel',
                                        shape=(v_in_dim, self.out_dim),
                                        initializer='glorot_normal')
    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10
    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = K.dot(q, self.q_kernel)
        kw = K.dot(k, self.k_kernel)
        vw = K.dot(v, self.v_kernel)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.nb_head, self.size_per_head))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.nb_head, self.size_per_head))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.nb_head, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.size_per_head**0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)

def position_id(x):
    if isinstance(x, list) and len(x) == 2:
        x, r = x
    else:
        r = 0
    pid = K.arange(K.shape(x)[1])
    pid = K.expand_dims(pid, 0)
    pid = K.tile(pid, [K.shape(x)[0], 1])
    return K.abs(pid - K.cast(r, 'int32'))



t1_in = Input(shape=(None,))#没有向量化的原始序列
t2_in = Input(shape=(None, word_size))#向量化的序列
s1_in = Input(shape=(None,))
s2_in = Input(shape=(None,))

t1,t2,s1, s2=t1_in,t2_in,s1_in,s2_in

mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(t1)

pid = Lambda(position_id)(t1)
position_embedding = Embedding(maxlen,word_size, embeddings_initializer='zeros')
pv = position_embedding(pid)

t2 = Dense(word_size, use_bias=False)(t2)

t = Add()([t2, pv])
t = Dropout(0.25)(t)
t = Lambda(lambda x: x[0] * x[1])([t, mask])

t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 2)
t = dilated_gated_conv1d(t, mask, 5)
t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 2)
t = dilated_gated_conv1d(t, mask, 5)
t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 2)
t = dilated_gated_conv1d(t, mask, 5)
t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 1)
t = dilated_gated_conv1d(t, mask, 1)

pn1 = Dense(char_size, activation='relu')(t)
pn1 = Dense(1, activation='sigmoid')(pn1)
pn2 = Dense(char_size, activation='relu')(t)
pn2 = Dense(1, activation='sigmoid')(pn2)


h = Attention(8, 16)([t, t, t, mask])
h = Concatenate()([t, h])
h = Conv1D(char_size, 3, activation='relu', padding='same')(h)
ps1 = Dense(1, activation='sigmoid')(h)
ps2 = Dense(1, activation='sigmoid')(h)
ps1 = Lambda(lambda x: x[0] * x[1])([ps1, pn1])
ps2 = Lambda(lambda x: x[0] * x[1])([ps2, pn2])



s1 = K.expand_dims(s1, 2)
s2 = K.expand_dims(s2, 2)

s1_loss = K.binary_crossentropy(s1, ps1)
s1_loss = K.sum(s1_loss * mask) / K.sum(mask)
s2_loss = K.binary_crossentropy(s2, ps2)
s2_loss = K.sum(s2_loss * mask) / K.sum(mask)
loss = (s1_loss + s2_loss)

subject_model = Model([t1_in, t2_in,s1_in, s2_in], [ps1, ps2])
subject_model.add_loss(loss)
subject_model.compile(optimizer=Adam(1e-3))
subject_model.summary()

class ExponentialMovingAverage:
    """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    """
    def __init__(self, model, momentum=0.9999):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]
    def inject(self):
        """添加更新算子到model.metrics_updates。
        """
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = K.moving_average_update(w1, w2, self.momentum)
            self.model.metrics_updates.append(op)
    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        K.batch_set_value(zip(self.ema_weights, self.old_weights))
    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        ema_weights = K.batch_get_value(self.ema_weights)
        K.batch_set_value(zip(self.model.weights, ema_weights))
    def reset_old_weights(self):
        """恢复模型到旧权重。
        """
        K.batch_set_value(zip(self.model.weights, self.old_weights))


EMAer = ExponentialMovingAverage(subject_model)
EMAer.inject()