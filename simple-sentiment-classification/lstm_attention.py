from keras.layers.core import Activation,Dense,Dropout,SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils import multi_gpu_model
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
import jieba
import xlwt
from keras.preprocessing.text import Tokenizer
import xlrd
from gensim.models import Word2Vec

from keras.layers import *

import keras.backend as K





def to_mask(x, mask, mode='mul'):

    """通用mask函数

    这里的mask.shape=[batch_size, seq_len]或[batch_size, seq_len, 1]

    """

    if mask is None:

        return x

    else:

        for _ in range(K.ndim(x) - K.ndim(mask)):

            mask = K.expand_dims(mask, K.ndim(mask))

        if mode == 'mul':

            return x * mask

        else:

            return x - (1 - mask) * 1e10





def extract_seq_patches(x, kernel_size, rate):

    """x.shape = [None, seq_len, seq_dim]

    滑动地把每个窗口的x取出来，为做局部attention作准备。

    """

    seq_dim = K.int_shape(x)[-1]

    seq_len = K.shape(x)[1]

    k_size = kernel_size + (rate - 1) * (kernel_size - 1)

    p_right = (k_size - 1) // 2

    p_left = k_size - 1 - p_right

    x = K.temporal_padding(x, (p_left, p_right))

    xs = [x[:, i: i + seq_len] for i in range(0, k_size, rate)]

    x = K.concatenate(xs, 2)

    return K.reshape(x, (-1, seq_len, kernel_size, seq_dim))





class OurLayer(Layer):

    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层

    """

    def reuse(self, layer, *args, **kwargs):

        if not layer.built:

            if len(args) > 0:

                inputs = args[0]

            else:

                inputs = kwargs['inputs']

            if isinstance(inputs, list):

                input_shape = [K.int_shape(x) for x in inputs]

            else:

                input_shape = K.int_shape(inputs)

            layer.build(input_shape)

        outputs = layer.call(*args, **kwargs)

        for w in layer.trainable_weights:

            if w not in self._trainable_weights:

                self._trainable_weights.append(w)

        for w in layer.non_trainable_weights:

            if w not in self._non_trainable_weights:

                self._non_trainable_weights.append(w)

        for u in layer.updates:

            if not hasattr(self, '_updates'):

                self._updates = []

            if u not in self._updates:

                self._updates.append(u)

        return outputs





class Attention(OurLayer):

    """多头注意力机制

    """

    def __init__(self, heads, size_per_head, key_size=None,

                 mask_right=False, **kwargs):

        super(Attention, self).__init__(**kwargs)

        self.heads = heads

        self.size_per_head = size_per_head

        self.out_dim = heads * size_per_head

        self.key_size = key_size if key_size else size_per_head

        self.mask_right = mask_right

    def build(self, input_shape):

        super(Attention, self).build(input_shape)

        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)

        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)

        self.v_dense = Dense(self.out_dim, use_bias=False)

    def call(self, inputs):

        q, k, v = inputs[: 3]

        v_mask, q_mask = None, None

        # 这里的mask.shape=[batch_size, seq_len]或[batch_size, seq_len, 1]

        if len(inputs) > 3:

            v_mask = inputs[3]

            if len(inputs) > 4:

                q_mask = inputs[4]

        # 线性变换

        qw = self.reuse(self.q_dense, q)

        kw = self.reuse(self.k_dense, k)

        vw = self.reuse(self.v_dense, v)

        # 形状变换

        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))

        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))

        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))

        # 维度置换

        qw = K.permute_dimensions(qw, (0, 2, 1, 3))

        kw = K.permute_dimensions(kw, (0, 2, 1, 3))

        vw = K.permute_dimensions(vw, (0, 2, 1, 3))

        # Attention

        a = K.batch_dot(qw, kw, [3, 3]) / self.key_size**0.5

        a = K.permute_dimensions(a, (0, 3, 2, 1))

        a = to_mask(a, v_mask, 'add')

        a = K.permute_dimensions(a, (0, 3, 2, 1))

        if (self.mask_right is not False) or (self.mask_right is not None):

            if self.mask_right is True:

                ones = K.ones_like(a[: 1, : 1])

                mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10

                a = a - mask

            else:

                # 这种情况下，mask_right是外部传入的0/1矩阵，shape=[q_len, k_len]

                mask = (1 - K.constant(self.mask_right)) * 1e10

                mask = K.expand_dims(K.expand_dims(mask, 0), 0)

                self.mask = mask

                a = a - mask

        a = K.softmax(a)

        self.a = a

        # 完成输出

        o = K.batch_dot(a, vw, [3, 2])

        o = K.permute_dimensions(o, (0, 2, 1, 3))

        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))

        o = to_mask(o, q_mask, 'mul')

        return o

    def compute_output_shape(self, input_shape):

        return (input_shape[0][0], input_shape[0][1], self.out_dim)





class SelfAttention(OurLayer):

    """多头自注意力机制

    """

    def __init__(self, heads, size_per_head, key_size=None,

                 mask_right=False, **kwargs):

        super(SelfAttention, self).__init__(**kwargs)

        self.heads = heads

        self.size_per_head = size_per_head

        self.out_dim = heads * size_per_head

        self.key_size = key_size if key_size else size_per_head

        self.mask_right = mask_right

    def build(self, input_shape):

        super(SelfAttention, self).build(input_shape)

        self.attention = Attention(

            self.heads,

            self.size_per_head,

            self.key_size,

            self.mask_right

        )

    def call(self, inputs):

        if isinstance(inputs, list):

            x, x_mask = inputs

            o = self.reuse(self.attention, [x, x, x, x_mask, x_mask])

        else:

            x = inputs

            o = self.reuse(self.attention, [x, x, x])

        return o

    def compute_output_shape(self, input_shape):

        if isinstance(input_shape, list):

            return (input_shape[0][0], input_shape[0][1], self.out_dim)

        else:

            return (input_shape[0], input_shape[1], self.out_dim)





class AtrousSelfAttention(OurLayer):

    """空洞多头自注意力机制

    说明：每个元素只跟相对距离为rate的倍数的元素有关联。

    """

    def __init__(self, heads, size_per_head, rate=1,

                 key_size=None, mask_right=False, **kwargs):

        super(AtrousSelfAttention, self).__init__(**kwargs)

        self.heads = heads

        self.size_per_head = size_per_head

        self.out_dim = heads * size_per_head

        self.key_size = key_size if key_size else size_per_head

        self.rate = rate

        self.mask_right = mask_right

    def build(self, input_shape):

        super(AtrousSelfAttention, self).build(input_shape)

        self.attention = Attention(

            self.heads,

            self.size_per_head,

            self.key_size,

            self.mask_right

        )

    def call(self, inputs):

        if isinstance(inputs, list):

            x, x_mask = inputs

        else:

            x, x_mask = inputs, None

        seq_dim = K.int_shape(x)[-1]

        # 补足长度，保证可以reshape

        seq_len = K.shape(x)[1]

        pad_len = self.rate - seq_len % self.rate

        x = K.temporal_padding(x, (0, pad_len))

        if x_mask is not None:

            x_mask = K.temporal_padding(x_mask, (0, pad_len))

        new_seq_len = K.shape(x)[1]

        # 变换shape

        x = K.reshape(x, (-1, new_seq_len // self.rate, self.rate, seq_dim))

        x = K.permute_dimensions(x, (0, 2, 1, 3))

        x = K.reshape(x, (-1, new_seq_len // self.rate, seq_dim))

        if x_mask is not None:

            x_mask = K.reshape(x_mask, (-1, new_seq_len // self.rate, self.rate, 1))

            x_mask = K.permute_dimensions(x_mask, (0, 2, 1, 3))

            x_mask = K.reshape(x_mask, (-1, new_seq_len // self.rate, 1))

        # 做attention

        if x_mask is not None:

            x = self.reuse(self.attention, [x, x, x, x_mask, x_mask])

        else:

            x = self.reuse(self.attention, [x, x, x])

        # 恢复shape

        x = K.reshape(x, (-1, self.rate, new_seq_len // self.rate, self.out_dim))

        x = K.permute_dimensions(x, (0, 2, 1, 3))

        x = K.reshape(x, (-1, new_seq_len, self.out_dim))

        x = x[:, : - pad_len]

        return x

    def compute_output_shape(self, input_shape):

        if isinstance(input_shape, list):

            return (input_shape[0][0], input_shape[0][1], self.out_dim)

        else:

            return (input_shape[0], input_shape[1], self.out_dim)





class LocalSelfAttention(OurLayer):

    """局部多头自注意力机制

    说明：每个元素只跟相对距离不超过neighbors的元素有关联，这里的rate

    是真正的膨胀率（跟膨胀卷积一样），如果不了解可以忽略，默认为1就好。

    """

    def __init__(self, heads, size_per_head, neighbors=1, rate=1,

                 key_size=None, mask_right=False, **kwargs):

        super(LocalSelfAttention, self).__init__(**kwargs)

        self.heads = heads

        self.size_per_head = size_per_head

        self.out_dim = heads * size_per_head

        self.key_size = key_size if key_size else size_per_head

        self.neighbors = neighbors

        self.rate = rate

        self.mask_right = mask_right

    def build(self, input_shape):

        super(LocalSelfAttention, self).build(input_shape)

        if self.mask_right:

            mask_right = np.ones((1, 1 + 2 * self.neighbors))

            mask_right[:, - self.neighbors : ] = 0

        else:

            mask_right = self.mask_right

        self.attention = Attention(

            self.heads,

            self.size_per_head,

            self.key_size,

            mask_right

        )

    def call(self, inputs):

        if isinstance(inputs, list):

            x, x_mask = inputs

        else:

            x, x_mask = inputs, None

        # 提取局部特征

        kernel_size = 1 + 2 * self.neighbors

        xp = extract_seq_patches(x, kernel_size, self.rate)

        if x_mask is not None:

            xp_mask = extract_seq_patches(x_mask, kernel_size, self.rate)

        # 变换shape

        seq_len = K.shape(x)[1]

        seq_dim = K.int_shape(x)[-1]

        x = K.reshape(x, (-1, 1, seq_dim))

        xp = K.reshape(xp, (-1, kernel_size, seq_dim))

        if x_mask is not None:

            xp_mask = K.reshape(xp_mask, (-1, kernel_size, 1))

        # 做attention

        if x_mask is not None:

            x = self.reuse(self.attention, [x, xp, xp, xp_mask])

        else:

            x = self.reuse(self.attention, [x, xp, xp])

        # 恢复shape

        x = K.reshape(x, (-1, seq_len, self.out_dim))

        x = to_mask(x, x_mask, 'mul')

        return x

    def compute_output_shape(self, input_shape):

        if isinstance(input_shape, list):

            return (input_shape[0][0], input_shape[0][1], self.out_dim)

        else:

            return (input_shape[0], input_shape[1], self.out_dim)





class SparseSelfAttention(OurLayer):

    """稀疏多头自注意力机制

    来自文章《Generating Long Sequences with Sparse Transformers》

    说明：每个元素只跟相对距离为rate的倍数的元素、以及相对距离不超过rate的元素有关联。

    """

    def __init__(self, heads, size_per_head, rate=2,

                 key_size=None, mask_right=False, **kwargs):

        super(SparseSelfAttention, self).__init__(**kwargs)

        self.heads = heads

        self.size_per_head = size_per_head

        self.out_dim = heads * size_per_head

        self.key_size = key_size if key_size else size_per_head

        assert rate != 1, u'if rate=1, please use SelfAttention directly'

        self.rate = rate

        self.neighbors = rate - 1

        self.mask_right = mask_right

    def build(self, input_shape):

        super(SparseSelfAttention, self).build(input_shape)

        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)

        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)

        self.v_dense = Dense(self.out_dim, use_bias=False)

    def call(self, inputs):

        if isinstance(inputs, list):

            x, x_mask = inputs

        else:

            x, x_mask = inputs, None

        seq_dim = K.int_shape(x)[-1]

        # 补足长度，保证可以reshape

        seq_len = K.shape(x)[1]

        pad_len = self.rate - seq_len % self.rate

        x = K.temporal_padding(x, (0, pad_len))

        if x_mask is not None:

            x_mask = K.temporal_padding(x_mask, (0, pad_len))

        new_seq_len = K.shape(x)[1]

        x = K.reshape(x, (-1, new_seq_len, seq_dim)) # 经过padding后shape可能变为None，所以重新声明一下shape

        # 线性变换

        qw = self.reuse(self.q_dense, x)

        kw = self.reuse(self.k_dense, x)

        vw = self.reuse(self.v_dense, x)

        # 提取局部特征

        kernel_size = 1 + 2 * self.neighbors

        kwp = extract_seq_patches(kw, kernel_size, self.rate) # shape=[None, seq_len, kernel_size, out_dim]

        vwp = extract_seq_patches(vw, kernel_size, self.rate) # shape=[None, seq_len, kernel_size, out_dim]

        if x_mask is not None:

            xp_mask = extract_seq_patches(x_mask, kernel_size, self.rate)

        # 形状变换

        qw = K.reshape(qw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.key_size))

        kw = K.reshape(kw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.key_size))

        vw = K.reshape(vw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.size_per_head))

        kwp = K.reshape(kwp, (-1, new_seq_len // self.rate, self.rate, kernel_size, self.heads, self.key_size))

        vwp = K.reshape(vwp, (-1, new_seq_len // self.rate, self.rate, kernel_size, self.heads, self.size_per_head))

        if x_mask is not None:

            x_mask = K.reshape(x_mask, (-1, new_seq_len // self.rate, self.rate, 1, 1))

            xp_mask = K.reshape(xp_mask, (-1, new_seq_len // self.rate, self.rate, kernel_size, 1, 1))

        # 维度置换

        qw = K.permute_dimensions(qw, (0, 3, 2, 1, 4)) # shape=[None, heads, r, seq_len // r, size]

        kw = K.permute_dimensions(kw, (0, 3, 2, 1, 4))

        vw = K.permute_dimensions(vw, (0, 3, 2, 1, 4))

        qwp = K.expand_dims(qw, 4)

        kwp = K.permute_dimensions(kwp, (0, 4, 2, 1, 3, 5)) # shape=[None, heads, r, seq_len // r, kernel_size, out_dim]

        vwp = K.permute_dimensions(vwp, (0, 4, 2, 1, 3, 5))

        if x_mask is not None:

            x_mask = K.permute_dimensions(x_mask, (0, 3, 2, 1, 4))

            xp_mask = K.permute_dimensions(xp_mask, (0, 4, 2, 1, 3, 5))

        # Attention1

        a = K.batch_dot(qw, kw, [4, 4]) / self.key_size**0.5

        a = K.permute_dimensions(a, (0, 1, 2, 4, 3))

        a = to_mask(a, x_mask, 'add')

        a = K.permute_dimensions(a, (0, 1, 2, 4, 3))

        if self.mask_right:

            ones = K.ones_like(a[: 1, : 1, : 1])

            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10

            a = a - mask

        # Attention2

        ap = K.batch_dot(qwp, kwp, [5, 5]) / self.key_size**0.5

        ap = K.permute_dimensions(ap, (0, 1, 2, 3, 5, 4))

        if x_mask is not None:

            ap = to_mask(ap, xp_mask, 'add')

        ap = K.permute_dimensions(ap, (0, 1, 2, 3, 5, 4))

        if self.mask_right:

            mask = np.ones((1, kernel_size))

            mask[:, - self.neighbors : ] = 0

            mask = (1 - K.constant(mask)) * 1e10

            for _ in range(4):

                mask = K.expand_dims(mask, 0)

            ap = ap - mask

        ap = ap[..., 0, :]

        # 合并两个Attention

        A = K.concatenate([a, ap], -1)

        A = K.softmax(A)

        a, ap = A[..., : K.shape(a)[-1]], A[..., K.shape(a)[-1] : ]

        # 完成输出1

        o1 = K.batch_dot(a, vw, [4, 3])

        # 完成输出2

        ap = K.expand_dims(ap, -2)

        o2 = K.batch_dot(ap, vwp, [5, 4])

        o2 = o2[..., 0, :]

        # 完成输出

        o = o1 + o2

        o = to_mask(o, x_mask, 'mul')

        o = K.permute_dimensions(o, (0, 3, 2, 1, 4))

        o = K.reshape(o, (-1, new_seq_len, self.out_dim))

        o = o[:, : - pad_len]

        return o

    def compute_output_shape(self, input_shape):

        if isinstance(input_shape, list):

            return (input_shape[0][0], input_shape[0][1], self.out_dim)

        else:

            return (input_shape[0], input_shape[1], self.out_dim)





class TrainablePositionEmbedding(OurLayer):

    """定义位置Embedding，直接训练出来

    """

    def __init__(self, maxlen, v_dim,

                 merge_mode='add', **kwargs):

        super(TrainablePositionEmbedding, self).__init__(**kwargs)

        self.maxlen = maxlen

        self.v_dim = v_dim

        self.merge_mode = merge_mode

    def build(self, input_shape):

        super(TrainablePositionEmbedding, self).build(input_shape)

        self.embeddings = self.add_weight(

            name='embeddings',

            shape=(self.maxlen, self.v_dim),

            initializer='zeros'

        )

    def call(self, inputs):

        """允许传入r（当前位置id）来得到相对位置向量

        """

        if isinstance(inputs, list):

            x, r = inputs

        else:

            x, r = inputs, 0

        pid = K.arange(K.shape(x)[1])

        pid = K.expand_dims(pid, 0)

        pid = K.tile(pid, [K.shape(x)[0], 1])

        pid = K.abs(pid - K.cast(r, 'int32'))

        pv = K.gather(self.embeddings, pid)

        if self.merge_mode == 'add':

            return pv + x

        else:

            return K.concatenate([x, pv])

    def compute_output_shape(self, input_shape):

        if self.merge_mode == 'add':

            return input_shape

        else:

            return (input_shape[0], input_shape[1], input_shape[2] + self.v_dim)





class SinCosPositionEmbedding(Layer):

    """Google提出来的Sin-Cos形式的位置Embedding

    """

    def __init__(self, v_dim,

                 merge_mode='add', **kwargs):

        super(SinCosPositionEmbedding, self).__init__(**kwargs)

        self.v_dim = v_dim

        self.merge_mode = merge_mode

    def call(self, inputs):

        """允许传入r（当前位置id）来得到相对位置向量

        """

        if isinstance(inputs, list):

            x, r = inputs

        else:

            x, r = inputs, 0

        pid = K.arange(K.shape(x)[1])

        pid = K.expand_dims(pid, 0)

        pid = K.tile(pid, [K.shape(x)[0], 1])

        pid = K.abs(pid - K.cast(r, 'int32'))

        pv = self.idx2pos(pid)

        if self.merge_mode == 'add':

            return pv + x

        else:

            return K.concatenate([x, pv])

    def idx2pos(self, pid):

        pid = K.cast(pid, 'float32')

        pid = K.expand_dims(pid, 2)

        pj = 1. / K.pow(10000., 2. / self.v_dim * K.arange(self.v_dim // 2, dtype='float32'))

        pj = K.expand_dims(pj, 0)

        pv = K.dot(pid, pj)

        pv1, pv2 = K.sin(pv), K.cos(pv)

        pv1, pv2 = K.expand_dims(pv1, 3), K.expand_dims(pv2, 3)

        pv = K.concatenate([pv1, pv2], 3)

        return K.reshape(pv, (K.shape(pv)[0], K.shape(pv)[1], self.v_dim))

    def compute_output_shape(self, input_shape):

        if self.merge_mode == 'add':

            return input_shape

        else:

            return input_shape[:-1] + (input_shape[-1] + self.v_dim,)


from keras.layers.core import Activation,Dense,Dropout,SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils import multi_gpu_model
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
import jieba
import xlwt
from keras.preprocessing.text import Tokenizer
import xlrd
from gensim.models import Word2Vec
import gensim
def step1_combine():
    f1=xlwt.Workbook()
    f2=xlwt.Workbook()
    sheetf1=f1.add_sheet("合并",cell_overwrite_ok=True)
    sheetf2=f2.add_sheet("合并",cell_overwrite_ok=True)
    r=xlrd.open_workbook(filename="combine.xls")
    sheet2=r.sheet_by_index(0)
    intag=sheet2.col_values(1)
    inwords=sheet2.col_values(2)
    inclass=sheet2.col_values(0)
    print(len(intag))
    print(len(inwords))
    print(len(inclass))
    count1=0
    ramdom1=8
    ramdom2=4
    count2=0
    
    for i in range(0,len(intag)):
        if i==ramdom1:
            ramdom1+=10
            sheetf2.write(count2,0,inclass[i])
            sheetf2.write(count2,1,intag[i])
            sheetf2.write(count2,2,inwords[i])
            count2+=1
        elif i==ramdom2:
            ramdom2+=10
            sheetf2.write(count2,0,inclass[i])
            sheetf2.write(count2,1,intag[i])
            sheetf2.write(count2,2,inwords[i])
            count2+=1
        else:
            sheetf1.write(count1,0,inclass[i])
            sheetf1.write(count1,1,intag[i])
            sheetf1.write(count1,2,inwords[i])
            count1=count1+1
    sheetf2.write(0,4,"总数")
    sheetf1.write(0,4,"总数")
    sheetf2.write(1,4,count2)
    sheetf1.write(1,4,count1)
    f1.save("combine_train.xls")
    f2.save("combine_test.xls")


def step2_pre_processing(suffix_stop="",suffix_start=""):
    r=xlrd.open_workbook(filename="combine_"+suffix_start+".xls")
    sheet1=r.sheet_by_index(0)
    intag=sheet1.col_values(1)
    inclass=sheet1.col_values(0)
    inwords=sheet1.col_values(2)
    print(len(intag))
    print(len(inwords))
    stop_word=[]
    with open("stop_word"+suffix_stop+".txt","r",encoding="GB18030") as file:
        instop_words=file.readlines()
        for i in instop_words:
           stop_word.append(i.strip())
    w=xlwt.Workbook()
    sheet2=w.add_sheet("准备文件",cell_overwrite_ok=True)
    count=0
    word_freqs=collections.Counter()
    vocall=[]
    inword=[]
    for word in inwords:
        word=jieba.cut(word)
        for i in word:
            word_freqs[i]+=1
    for word in inwords:
        word=jieba.cut(word)
        new_word=""
        for i in word:
            if word_freqs[i]>2:
                if i not in vocall:
                    vocall.append(i)
                if i not in stop_word:
                    new_word=new_word+i+' '
        new_word=new_word.strip()
        inword.append(new_word)
        sheet2.write(count,2,new_word)
        count=count+1
    print(count)
    print(len(intag))
    maxlen=0
    sumlen=0
    esumlen=0
    num=0
    num=len(inword)
    for i in inword:
        sumlen=sumlen+len(i)
        if len(i)>maxlen:
            maxlen=len(i)
    esumlen=sumlen/num
    outclass=[]
    outclass_count=collections.Counter()
    intag_count=collections.Counter(intag)
    n=6
    sheet2.write(0,11+n,"总个数")
    sheet2.write(0,10+n,"平均长度")
    sheet2.write(0,9+n,"最长长度")
    sheet2.write(1,11+n,num)
    sheet2.write(1,10+n,esumlen)
    sheet2.write(1,9+n,maxlen)
    sheet2.write(0,8+n,"分割线")
    sheet2.write(0,6+n,"情感")
    sheet2.write(1,6+n,"1")
    sheet2.write(2,6+n,"0")
    sheet2.write(0,7+n,"数量")
    sheet2.write(0,6,"词总数")
    sheet2.write(1,6,len(vocall))
    for i in range(0,len(vocall)):
        sheet2.write(i,4,vocall[i])
        sheet2.write(i,5,word_freqs[vocall[i]])
    if intag_count["1"]==0:
        sheet2.write(1,7+n,intag_count[1])
        sheet2.write(2,7+n,intag_count[0])
    else:
        sheet2.write(1,7+n,intag_count["1"])
        sheet2.write(2,7+n,intag_count["0"])
    for i in range(0,len(intag)):
        sheet2.write(i,1,intag[i])
    for i in range(0,len(inclass)):
        sheet2.write(i,0,inclass[i])
        outclass_count[inclass[i]]+=1
        if inclass[i] not in outclass:
            outclass.append(inclass[i])
    for i in range(0,len(stop_word)):
        sheet2.write(i,3,stop_word[i])
    sheet2.write(0,4+n,"种类")
    sheet2.write(0,5+n,"数量")
    for i in range(0,len(outclass)):
        sheet2.write(i+1,4+n,outclass[i])
        sheet2.write(i+1,5+n,outclass_count[outclass[i]])
    w.save("pre处理"+suffix_stop+suffix_start+".xls")
def spilt_words(suffix=""):
    r=xlrd.open_workbook(filename="pre处理"+suffix+".xls")
    sheet1=r.sheet_by_index(0)
    intag=sheet1.col_values(1)
    inwords=sheet1.col_values(2)
    inclass=sheet1.col_values(0)
    elen=sheet1.col_values(16)
    elen=elen[1]
    elen=int(elen)
    print(elen)
    out1_words=[]
    out2_words=[]
    out1_tag=[]
    out2_tag=[]
    out1_class=[]
    out2_class=[]
    maxlen=elen
    ex=0
    ex_tag=[]
    for i,precess_words in enumerate(inwords):
        precess_words=inwords[i]
        swich=0
        while len(precess_words)>maxlen:
            if swich==0:
                mid=precess_words[0:maxlen]
                out1_words.append(mid)
                out1_tag.append(intag[i])
                out1_class.append(inclass[i])
                out2_words.append(mid)
                out2_tag.append(intag[i])
                out2_class.append(inclass[i])
                ex_tag.append(ex)
                swich=1
            else:
                mid=precess_words[0:maxlen]
                out2_words.append(mid)
                out2_tag.append(intag[i])
                out2_class.append(inclass[i])
                ex_tag.append(ex)
            precess_words=precess_words[maxlen:len(precess_words)].strip()
            
        if len(precess_words)!=0:
            if swich==0:
                out1_words.append(precess_words)
                out2_words.append(precess_words)
                out1_tag.append(intag[i])
                out2_tag.append(intag[i])
                out1_class.append(inclass[i])
                out2_class.append(inclass[i])
                ex_tag.append(ex)
            elif len(precess_words)>maxlen/4:
                out2_words.append(precess_words)
                out2_tag.append(intag[i])
                out2_class.append(inclass[i])
                ex_tag.append(ex)
        ex+=1
    print("start save.........")
    w=xlwt.Workbook()
    sheet2=w.add_sheet("准备文件",cell_overwrite_ok=True)
    sheet2.write(0,14,maxlen)
    sheet2.write(0,11,"切割丢掉")
    sheet2.write(1,11,len(out1_tag))
    sheet2.write(0,12,"切割扩充")
    sheet2.write(1,12,len(out2_tag))
    swich=0
    for i in range(0,len(out2_words)):
        if i==65000:
            swich+=1
            break
        sheet2.write(i,3,out2_class[i])
        sheet2.write(i,4,out2_tag[i])
        sheet2.write(i,5,out2_words[i])
        sheet2.write(i,6,ex_tag[i])
    if swich==1:
        sheet2.write(0,13,1)
        for i in range(65000,len(out2_words)):
            sheet2.write(i-65000,7,out2_class[i])
            sheet2.write(i-65000,8,out2_tag[i])
            sheet2.write(i-65000,9,out2_words[i])
            sheet2.write(i-65000,10,ex_tag[i])
    else:
        sheet2.write(0,13,0)
    for i in range(0,len(out1_words)):
        sheet2.write(i,0,out1_class[i])
        sheet2.write(i,1,out1_tag[i])
        sheet2.write(i,2,out1_words[i])
    w.save("切割"+suffix+".xls")
def model(suffix=""):
    for round in range(0,2):
        rtest=xlrd.open_workbook(filename="切割"+suffix+"test.xls")
        rtrain=xlrd.open_workbook(filename="切割"+suffix+"train.xls")
        r_vocall1=xlrd.open_workbook(filename="pre处理"+suffix+"test.xls")
        r_vocall2=xlrd.open_workbook(filename="pre处理"+suffix+"train.xls")
        sheet_test=rtest.sheet_by_index(0)
        sheet_train=rtrain.sheet_by_index(0)
        sheet1_vocall=r_vocall1.sheet_by_index(0)
        sheet2_vocall=r_vocall2.sheet_by_index(0)
        invocal1=sheet1_vocall.col_values(4)
        
        invocal2=sheet2_vocall.col_values(4)
        for i in range(0,len(invocal1)):
            if len(invocal1[i])==0:
                invocall=invocal1[:i]
                print("1")
                break
        
        for i in range(0,len(invocal2)):
            if len(invocal2[i])==0:
                print("1")
                invocal2=invocal2[:i]
                break
        for i in invocal2:
            if i not in invocall:
                invocall.append(i)
        print(len(invocall))
        vocall_size=len(invocall)
        if round==1:
            ex_tag=sheet_test.col_values(6)
        xtrain=sheet_train.col_values(2+round*3)
        ztrain=sheet_train.col_values(0+round*3)
        ytrain=sheet_train.col_values(1+round*3)
        xtest=sheet_test.col_values(2+round*3)
        ztest=sheet_test.col_values(0+round*3)
        ytest=sheet_test.col_values(1+round*3)
        
        for i in range(0,len(xtrain)):
            if len(xtrain[i])==0:
                xtrain=xtrain[:i]
                ztrain=ztrain[:i]
                ytrain=ytrain[:i]
                break
        for i in range(0,len(xtest)):
            if len(xtest[i])==0:
                xtest=xtest[:i]
                ytest=ytest[:i]
                ztest=ztest[:i]
                break
        print(round*3)
        print(len(xtrain),"xtrain")
        print(len(ytrain),"ytrain")
        print(len(xtest),"xtest")
        print(len(ytest),"ytest")
        if round==1:
            other=sheet_train.cell(0,13).value
            other=int(other)
            print(other)
            if other==1:
                xtrain=xtrain+sheet_train.col_values(9)
                ytrain=ytrain+sheet_train.col_values(8)
                ztrain=ztrain+sheet_train.col_values(7)
                for i in range(0,len(xtrain)):
                    if len(xtrain[i])==0:
                        xtrain=xtrain[:i]
                        ztrain=ztrain[:i]
                        ytrain=ytrain[:i]
                        break

        tokenizer=Tokenizer(num_words=vocall_size)
        tokenizer.fit_on_texts(invocall)
        xtrain=tokenizer.texts_to_sequences(xtrain)
        xtest=tokenizer.texts_to_sequences(xtest)
        maxlen=0
        for i in xtrain:
            if len(i)>maxlen:
                maxlen=len(i)
        for i in xtest:
            if len(i)>maxlen:
                maxlen=len(i)
        print(maxlen,"maxlen")
        xtrain=pad_sequences(xtrain,padding='post',maxlen=maxlen)
        xtest=pad_sequences(xtest,padding='post',maxlen=maxlen)
        print(len(ytrain),len(xtrain))
        print(len(ytest),len(xtest))
        for i in range(0,len(ytrain)):
            ytrain[i]=int(ytrain[i])
        for i in range(0,len(ytest)):
            ytest[i]=int(ytest[i])
        embedding_size=150
        hidden_layer_size=64
        batch_size=128
        num_epochs=3
        model=Sequential()
        model.add(Embedding(vocall_size,embedding_size,input_length=maxlen))
        model.add(SpatialDropout1D(0.2))
        model.add(Attention(1,1))
        model.add(LSTM(hidden_layer_size,dropout=0.2,recurrent_dropout=0.2))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        model.summary()
        
        history=model.fit(xtrain,ytrain,epochs=7,batch_size=64)
        loss,accuracy=model.evaluate(xtest,ytest)
        print(loss,accuracy)
        """
        plt.subplot(211)
        plt.title("Accuracy"+suffix)
        plt.plot(history.history['acc'],color="g",label="Train")
      
        plt.legend(loc="best")

        plt.subplot(212)
        plt.title("Loss")
        plt.plot(history.history['loss'],color="g",label="Train")
       
        plt.legend(loc="best")

        plt.tight_layout()
        plt.show()
        """

        w=xlwt.Workbook()
        sheet2=w.add_sheet("准备文件",cell_overwrite_ok=True)
        sheet2.write(0,0,"predict")
        sheet2.write(0,1,"ytest")
        sheet2.write(0,2,"xtest")
        sheet2.write(0,3,"ex_tag")
        sheet2.write(0,4,"loss")
        sheet2.write(1,4,loss)
        sheet2.write(0,5,"acc")
        sheet2.write(1,5,accuracy)
        ypred=model.predict_classes(xtest,1)
        xtest=tokenizer.sequences_to_texts(xtest)
        for index in range(0,len(ypred)):
            sheet2.write(1+index,0,int(ypred[index][0]))
            sheet2.write(1+index,1,ytest[index])
            sheet2.write(1+index,2,xtest[index])
            if round==1:
                sheet2.write(1+index,3,ex_tag[index])
        if round==1:
            ac_sum=[]
            for i in range(0,len(ypred)):
                ac=0
                count=1
                ac=ac+int(ypred[i][0])
                for i2 in range(i,len(ypred)):
                    if i!=len(ypred)-1 and ex_tag[i]==ex_tag[i+1]:
                        i=i+1
                        ac=ac+int(ypred[i][0])
                        count=count+1
                    else:
                        acca=ac/count
                        if acca>=0.5:
                            ac_sum.append("1")
                        else:
                            ac_sum.append("0")
                        break
            right=0
            refer_tag=sheet_test.col_values(1)
            for i in range(0,len(ac_sum)):
                if refer_tag[i]==ac_sum[i]:
                    right+=1
            acr=right/len(ac_sum)
            sheet2.write(0,7,"acc")
            sheet2.write(1,7,acr)
        if round==0:
            w.save("result切割"+suffix+".xls")
        else:
            w.save("result扩充"+suffix+".xls")

def model_word2vec(suffix=""):
    for round in range(0,2):
        rtest=xlrd.open_workbook(filename="切割"+suffix+"test.xls")
        rtrain=xlrd.open_workbook(filename="切割"+suffix+"train.xls")
        r_vocall1=xlrd.open_workbook(filename="pre处理"+suffix+"test.xls")
        r_vocall2=xlrd.open_workbook(filename="pre处理"+suffix+"train.xls")
        sheet_test=rtest.sheet_by_index(0)
        sheet_train=rtrain.sheet_by_index(0)
        sheet1_vocall=r_vocall1.sheet_by_index(0)
        sheet2_vocall=r_vocall2.sheet_by_index(0)
        invocal1=sheet1_vocall.col_values(4)
        
        invocal2=sheet2_vocall.col_values(4)
        for i in range(0,len(invocal1)):
            if len(invocal1[i])==0:
                invocall=invocal1[:i]
                print("1")
                break
        
        for i in range(0,len(invocal2)):
            if len(invocal2[i])==0:
                print("1")
                invocal2=invocal2[:i]
                break
        for i in invocal2:
            if i not in invocall:
                invocall.append(i)
        print(len(invocall))
        vocall_size=len(invocall)
        if round==1:
            ex_tag=sheet_test.col_values(6)
        xtrain=sheet_train.col_values(2+round*3)
        ztrain=sheet_train.col_values(0+round*3)
        ytrain=sheet_train.col_values(1+round*3)
        xtest=sheet_test.col_values(2+round*3)
        ztest=sheet_test.col_values(0+round*3)
        ytest=sheet_test.col_values(1+round*3)
        
        for i in range(0,len(xtrain)):
            if len(xtrain[i])==0:
                xtrain=xtrain[:i]
                ztrain=ztrain[:i]
                ytrain=ytrain[:i]
                break
        for i in range(0,len(xtest)):
            if len(xtest[i])==0:
                xtest=xtest[:i]
                ytest=ytest[:i]
                ztest=ztest[:i]
                break
        print(round*3)
        print(len(xtrain),"xtrain")
        print(len(ytrain),"ytrain")
        print(len(xtest),"xtest")
        print(len(ytest),"ytest")
        if round==1:
            other=sheet_train.cell(0,13).value
            other=int(other)
            print(other)
            if other==1:
                xtrain=xtrain+sheet_train.col_values(9)
                ytrain=ytrain+sheet_train.col_values(8)
                ztrain=ztrain+sheet_train.col_values(7)

        tokenizer=Tokenizer(num_words=vocall_size)
        tokenizer.fit_on_texts(invocall)
        xtrain=tokenizer.texts_to_sequences(xtrain)
        xtest=tokenizer.texts_to_sequences(xtest)
        maxlen=0
        for i in xtrain:
            if len(i)>maxlen:
                maxlen=len(i)
        for i in xtest:
            if len(i)>maxlen:
                maxlen=len(i)
        print(maxlen,"maxlen")
        xtrain=pad_sequences(xtrain,padding='post',maxlen=maxlen)
        xtest=pad_sequences(xtest,padding='post',maxlen=maxlen)
        print(len(ytrain),len(xtrain))
        print(len(ytest),len(xtest))
        for i in range(0,len(ytrain)):
            ytrain[i]=int(ytrain[i])
        for i in range(0,len(ytest)):
            ytest[i]=int(ytest[i])
        modelw2v = gensim.models.Word2Vec.load("word2vec_150_lstm.model")
        embedding_matrix = np.zeros(shape=(vocall_size + 1,150))
        for word, i in invocall():
            embedding_vector = modelw2v[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        embedding_size=150
        hidden_layer_size=64
        batch_size=128
        num_epochs=3
        model=Sequential()
        model.add(Embedding(vocall_size+1,embedding_size,trainable=False,weights=[embedding_matrix],input_length=maxlen))     
        model.add(SpatialDropout1D(0.2))
        model.add(Attention())
        model.add(LSTM(hidden_layer_size,dropout=0.2,recurrent_dropout=0.2))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        model.summary()
        
        history=model.fit(xtrain,ytrain,epochs=7+round*5,batch_size=64)
        loss,accuracy=model.evaluate(xtest,ytest)
        print(loss,accuracy)
        """
        plt.subplot(211)
        plt.title("Accuracy"+suffix)
        plt.plot(history.history['acc'],color="g",label="Train")
      
        plt.legend(loc="best")

        plt.subplot(212)
        plt.title("Loss")
        plt.plot(history.history['loss'],color="g",label="Train")
       
        plt.legend(loc="best")

        plt.tight_layout()
        plt.show()
        """

        w=xlwt.Workbook()
        sheet2=w.add_sheet("准备文件",cell_overwrite_ok=True)
        sheet2.write(0,0,"predict")
        sheet2.write(0,1,"ytest")
        sheet2.write(0,2,"xtest")
        sheet2.write(0,3,"ex_tag")
        sheet2.write(0,4,"loss")
        sheet2.write(1,4,loss)
        sheet2.write(0,5,"acc")
        sheet2.write(1,5,accuracy)
        ypred=model.predict_classes(xtest,1)
        xtest=tokenizer.sequences_to_texts(xtest)
        for index in range(0,len(ypred)):
            sheet2.write(1+index,0,int(ypred[index][0]))
            sheet2.write(1+index,1,ytest[index])
            sheet2.write(1+index,2,xtest[index])
            if round==1:
                sheet2.write(1+index,3,ex_tag[index])
        if round==1:
            ac_sum=[]
            for i in range(0,len(ypred)):
                ac=0
                count=1
                ac=ac+int(ypred[i][0])
                for i2 in range(i,len(ypred)):
                    if i!=len(ypred)-1 and ex_tag[i]==ex_tag[i+1]:
                        i=i+1
                        ac=ac+int(ypred[i][0])
                        count=count+1
                    else:
                        acca=ac/count
                        if acca>=0.5:
                            ac_sum.append("1")
                        else:
                            ac_sum.append("0")
                        break
            right=0
            refer_tag=sheet_test.col_values(1)
            for i in range(0,len(ac_sum)):
                if refer_tag[i]==ac_sum[i]:
                    right+=1
            acr=right/len(ac_sum)
            sheet2.write(0,7,"acc")
            sheet2.write(1,7,acr)
        if round==0:
            w.save("result切割"+suffix+"w2v.xls")
        else:
            w.save("result扩充"+suffix+"w2v.xls")

"""
step2_pre_processing(suffix_stop="3",suffix_start="train")
step2_pre_processing(suffix_stop="3",suffix_start="test")
spilt_words(suffix="3train")
spilt_words(suffix="3test")
"""

model("3")
model_word2vec("3")





      



