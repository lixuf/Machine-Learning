from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

#json格式
orders = ['term', 'polar']

#参数
word_size=150
maxlen=1000

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

t1,t2=t1_in,t2_in

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

subject_model = Model([t1_in, t2_in], [ps1, ps2])



class Evaluate(Callback):
    def __init__(self):
        self.F1 = []
        self.best = 0.
        self.passed = 0
        self.stage = 0
    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，不warmup有不收敛的可能。
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * 1e-3
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
    def on_epoch_end(self, epoch, logs=None):
        EMAer.apply_ema_weights()
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            train_model.save_weights('best_model.weights')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))
        EMAer.reset_old_weights()
        if epoch + 1 == 50 or (
            self.stage == 0 and epoch > 10 and
            (f1 < 0.5 or np.argmax(self.F1) < len(self.F1) - 8)
        ):
            self.stage = 1
            train_model.load_weights('best_model.weights')
            EMAer.initialize()
            K.set_value(self.model.optimizer.lr, 1e-4)
            K.set_value(self.model.optimizer.iterations, 0)
            opt_weights = K.batch_get_value(self.model.optimizer.weights)
            opt_weights = [w * 0. for w in opt_weights]
            K.batch_set_value(zip(self.model.optimizer.weights, opt_weights))
    def evaluate(self):
        A, B, C = 1e-10, 1e-10, 1e-10
        F = open('dev_pred.json', 'w')
        for d in tqdm(iter(dev_data)):#dev_data原始数据
            R = set(extract_items(d['text']))
            T = set(d['spo_list'])
            A += len(R & T)
            B += len(R)
            C += len(T)
            s = json.dumps({
                'text': d['text'],
                'spo_list': [
                    dict(zip(orders, spo)) for spo in T
                ],
                'spo_list_pred': [
                    dict(zip(orders, spo)) for spo in R
                ],
                'new': [
                    dict(zip(orders, spo)) for spo in R - T
                ],
                'lack': [
                    dict(zip(orders, spo)) for spo in T - R
                ]
            }, ensure_ascii=False, indent=4)
            F.write(s.encode('utf-8') + '\n')
        F.close()
        return 2 * A / (B + C), A / B, A / C
