

k=82
r=512
Batch_size=16
Epochs=500

import numpy as np
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.engine.topology import Layer
from keras import activations
import tensorflow as tf
import joblib
from bert4keras.optimizers import extend_with_parameter_wise_lr as l_bei
from bert4keras.optimizers import extend_with_piecewise_linear_lr as l_fenduan
from bert4keras.optimizers import extend_with_layer_adaptation as l_shiying
from bert4keras.layers import LayerNormalization as LN
from bert4keras.optimizers import AdaFactorV1
from keras.layers import ConvLSTM2D
#数据集
import pickle
from os import path


def load_data_part_a(domain=0):
	# [[train_s, train_mask, train_position], train_tags], [[test_s, test_mask, test_position], test_tags]]
	file_dir = r'C:\Users\TR\source\repos\CRF尝试-神经网络版\CRF尝试-神经网络版\utils\data'
	train_data = joblib.load(open(file_dir + '\dataset{}_partA_train'.format(domain), 'rb'))
	test_data = joblib.load(open(file_dir + '\dataset{}_partA_test'.format(domain), 'rb'))
	return train_data, test_data


for i in range(1):
	itrain_data, itest_data = load_data_part_a(domain=i)
	itrain_x, itag = itrain_data
	isentences, imask, iposition = itrain_x
	itest_x, ivtag = itest_data
	ivsentences, ivmask, ivposition = itest_x
	
	print(np.shape(itag))
	print(np.shape(ivtag))
	print(np.shape(isentences))
	print(np.shape(ivsentences))
	print(np.shape(imask))
	print(np.shape(ivmask))
	print(np.shape(iposition))
	print(np.shape(ivposition))

	if i==0:
		tag=itag 
		sentences=isentences
		mask=imask
		position=iposition
		vtag=ivtag
		vsentences=ivsentences
		vmask=ivmask
		vposition=ivposition
	else:
		tag=np.concatenate([tag,itag])
		sentences=np.concatenate([sentences,isentences])
		mask=np.concatenate([mask,imask])
		position=np.concatenate([position,iposition])
		vtag=np.concatenate([vtag,ivtag])
		vsentences=np.concatenate([vsentences,ivsentences])
		vmask=np.concatenate([vmask,ivmask])
		vposition=np.concatenate([vposition,ivposition])

print(np.shape(tag))
print(np.shape(vtag))
print(np.shape(sentences))
print(np.shape(vsentences))
print(np.shape(mask))
print(np.shape(vmask))
print(np.shape(position))
print(np.shape(vposition))

#各种预定义层和模型中需要的函数





def cumsoftmax(x, mode='l2r'):
    """先softmax，然后cumsum，
    cumsum区分从左到右、从右到左两种模式
    """
    axis = K.ndim(x) - 1
    if mode == 'l2r':
        x = K.softmax(x, axis=axis)
        x = K.cumsum(x, axis=axis)
        return x
    elif mode == 'r2l':
        x = x[..., ::-1]
        x = K.softmax(x, axis=axis)
        x = K.cumsum(x, axis=axis)
        return x[..., ::-1]
    else:
        return x

class ONLSTM(Layer):
    """实现有序LSTM，来自论文
    Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks
    """
    def __init__(self,
                 units,
                 levels,
                 return_sequences=False,
                 dropconnect=None,
                 **kwargs):
        assert units % levels == 0
        self.units = units
        self.levels = levels
        self.chunk_size = units // levels
        self.return_sequences = return_sequences
        self.dropconnect = dropconnect
        super(ONLSTM, self).__init__(**kwargs)
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4 + self.levels * 2),
            name='kernel',
            initializer='glorot_uniform')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4 + self.levels * 2),
            name='recurrent_kernel',
            initializer='orthogonal')
        self.bias = self.add_weight(
            shape=(self.units * 4 + self.levels * 2,),
            name='bias',
            initializer='zeros')
        self.built = True
        if self.dropconnect:
            self._kernel = K.dropout(self.kernel, self.dropconnect)
            self._kernel = K.in_train_phase(self._kernel, self.kernel)
            self._recurrent_kernel = K.dropout(self.recurrent_kernel, self.dropconnect)
            self._recurrent_kernel = K.in_train_phase(self._recurrent_kernel, self.recurrent_kernel)
        else:
            self._kernel = self.kernel
            self._recurrent_kernel = self.recurrent_kernel
    def one_step(self, inputs, states):
        x_in, (c_last, h_last) = inputs, states
        x_out = K.dot(x_in, self._kernel) + K.dot(h_last, self._recurrent_kernel)
        x_out = K.bias_add(x_out, self.bias)
        f_master_gate = cumsoftmax(x_out[:, :self.levels], 'l2r')
        f_master_gate = K.expand_dims(f_master_gate, 2)
        i_master_gate = cumsoftmax(x_out[:, self.levels: self.levels * 2], 'r2l')
        i_master_gate = K.expand_dims(i_master_gate, 2)
        x_out = x_out[:, self.levels * 2:]
        x_out = K.reshape(x_out, (-1, self.levels * 4, self.chunk_size))
        f_gate = K.sigmoid(x_out[:, :self.levels])
        i_gate = K.sigmoid(x_out[:, self.levels: self.levels * 2])
        o_gate = K.sigmoid(x_out[:, self.levels * 2: self.levels * 3])
        c_in = K.tanh(x_out[:, self.levels * 3:])
        c_last = K.reshape(c_last, (-1, self.levels, self.chunk_size))
        overlap = f_master_gate * i_master_gate
        c_out = overlap * (f_gate * c_last + i_gate * c_in) + \
                (f_master_gate - overlap) * c_last + \
                (i_master_gate - overlap) * c_in
        h_out = o_gate * K.tanh(c_out)
        c_out = K.reshape(c_out, (-1, self.units))
        h_out = K.reshape(h_out, (-1, self.units))
        out = K.concatenate([h_out, f_master_gate[..., 0], i_master_gate[..., 0]], 1)
        return out, [c_out, h_out]
    def call(self, inputs):
        initial_states = [
            K.zeros((K.shape(inputs)[0], self.units)),
            K.zeros((K.shape(inputs)[0], self.units))
        ] # 定义初始态(全零)
        outputs = K.rnn(self.one_step, inputs, initial_states)
        self.distance = 1 - K.mean(outputs[1][..., self.units: self.units + self.levels], -1)
        self.distance_in = K.mean(outputs[1][..., self.units + self.levels:], -1)
        if self.return_sequences:
            return outputs[1][..., :self.units]
        else:
            return outputs[0][..., :self.units]
    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.units)
        else:
            return (input_shape[0], self.units)


def add_noise_in_train(x,mask):
	return K.in_train_phase(x+K.random_normal(x)*mask)

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    return scale * x

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)

def maxy(a,d):
	#寻找y_pre[i1][i2][0],y_pre[i1][i2][1],y_pre[i1][i2][2]中最大值
	#若相等优先后面的
	if d[3]>0:
		return np.array([0,0,0,1])
	c=np.zeros(shape=(4,),dtype='int32')
	b=np.argmax(a[0:3])
	c[b]=1
	return c


def get_target_position(tags):
	ops = []
	for i in range(len(tags)):
		if tags[i] == 1:
			begin = i
			end = i + 1
			for j in range(i + 1, len(tags)):
				if tags[j] != 2:
					end = j
					break
			ops.append([begin, end])
	return ops
def evaluate_part_a(y, y_pre):
	for i1 in range(np.shape(y_pre)[0]):
		for i2 in range(k):
			y_pre[i1][i2]=maxy(y_pre[i1][i2],y[i1][i2])
	print(y_pre[2])
	#"""传进来array"""
	y = y.tolist()
	y_pre = y_pre.tolist()
	y = [[t.index(1) for t in s] for s in y]
	y_pre = [[t.index(1) for t in s] for s in y_pre]

	TP = 0
	TPFP = 0
	TPFN = 0
	for s, s_pre in zip(y, y_pre):
		ops = get_target_position(s)
		ops_pre = get_target_position(s_pre)
		for op in ops:
			if op in ops_pre:
				TP += 1
		TPFP += len(ops_pre)
		TPFN += len(ops)
	if TPFP==0:
		p=0
	else:
		p = TP / TPFP
	if TPFN==0:
		r=0
	else:
		r = TP / TPFN
	if p+r==0:
		f=0
	else:
		f = 2 * p * r / (p + r)
	return p, r, f

def evaluate(vsentences,vmask,vtag,vposition):#回调器中测试函数
	y_pred=model.predict([vsentences,vmask,vposition])
	print(np.shape(y_pred))
	return evaluate_part_a(vtag,y_pred)

def viterbi(nodes, trans): # viterbi算法
    paths = nodes[0] # 初始化起始路径
    for l in range(1, len(nodes)): # 遍历后面的节点
        paths_old,paths = paths,{}
        for n,ns in nodes[l].items(): # 当前时刻的所有节点
            max_path,max_score = '',-1e10
            for p,ps in paths_old.items(): # 截止至前一时刻的最优路径集合
                score = ns + ps + trans[p[-1]+n] # 计算新分数
                if score > max_score: # 如果新分数大于已有的最大分
                    max_path,max_score = p+n, score # 更新路径
            paths[max_path] = max_score # 储存到当前时刻所有节点的最优路径
    return max_in_dict(paths)

class CRF(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，
    而预测则需要另外建立模型。
    """
    def __init__(self, ignore_last_label=False, **kwargs):
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        """
        inputs, mask = inputs[:, :-1], inputs[:, -1:]
        states = K.expand_dims(states[0], 2)  # (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans, 0)  # (1, output_dim, output_dim)
        outputs = K.logsumexp(states + trans, 1)  # (batch_size, output_dim)
        outputs = outputs + inputs
        outputs = mask * outputs + (1 - mask) * states[:, :, 0]
        return outputs, [outputs]

    def path_score(self, inputs, labels):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
        """
        point_score = K.sum(K.sum(inputs * labels, 2), 1, keepdims=True)  # 逐标签得分
        labels1 = K.expand_dims(labels[:, :-1], 3)
        labels2 = K.expand_dims(labels[:, 1:], 2)
        labels = labels1 * labels2  # 两个错位labels，负责从转移矩阵中抽取目标转移得分
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        trans_score = K.sum(K.sum(trans * labels, [2, 3]), 1, keepdims=True)
        return point_score + trans_score  # 两部分得分之和

    def call(self, inputs):  # CRF本身不改变输出，它只是一个loss
        return inputs

    def loss(self, y_true, y_pred):  # 目标y_pred需要是one hot形式
        if self.ignore_last_label:
            mask = 1 - y_true[:, :, -1:]
        else:
            mask = K.ones_like(y_pred[:, :, :1])
        y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
        path_score = self.path_score(y_pred, y_true)  # 计算分子（对数）
        init_states = [y_pred[:, 0]]  # 初始状态
        y_pred = K.concatenate([y_pred, mask])
        log_norm, _, _ = K.rnn(self.log_norm_step, y_pred[:, 1:], init_states)  # 计算Z向量（对数）
        log_norm = K.logsumexp(log_norm, 1, keepdims=True)  # 计算Z（对数）
        return log_norm - path_score  # 即log(分子/分母)

    def accuracy(self, y_true, y_pred):  # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1 - y_true[:, :, -1] if self.ignore_last_label else None
        y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal * mask) / K.sum(mask)

class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:,:,:,0]) #shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = softmax(b, 1)
            # o = K.batch_dot(c, u_hat_vecs, [2, 2])
            o = tf.einsum('bin,binj->bij', c, u_hat_vecs)
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                o = K.l2_normalize(o, -1)
                # b = K.batch_dot(o, u_hat_vecs, [2, 3])
                b = tf.einsum('bij,binj->bin', o, u_hat_vecs)
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def model1():#模型主体1
	sen_in=Input(shape=(k,r))
	mask_in=Input(shape=(k,r))
	position_in=Input(shape=(k,r))
	sen,mas,positi=sen_in,mask_in,position_in

	#t=Add()([sen,positi])
	#t=Masking()(sen)
	#t=Dropout(0.25)(t)
	#t=Lambda(lambda x: x[0] * x[1])([t, mas])

	t=sen

	#t=Lambda(lambda x: x[0] * x[1])([t, mas])

	#t=Lambda(add_noise_in_train)(t,mas)
	#t=Lambda(lambda x: x[0] * x[1])([t, mas])





	#t=Lambda(lambda x: x[0] * x[1])([t, mas])

	t=Bidirectional(CuDNNGRU(150,return_sequences=True,name="gru1"))(t)
	t=Dropout(0.25)(t)



	t=Bidirectional(CuDNNGRU(150,return_sequences=True,name="gru2"))(t)
	t=Dropout(0.25)(t)

	

	#t=Lambda(lambda x: K.concatenate(x ,axis=-1))([t1,t2])
	t=Bidirectional(CuDNNGRU(150,return_sequences=True,name="gru3"))(t)
	t=Dropout(0.25)(t)







	t=Capsule(k,40,share_weights=False,name="cap")(t)
	t=Dropout(0.25)(t)

	t=Dense(4)(t)

	#t=Lambda(lambda x: x[0] * x[1])([t, mas[:,:,:4]])

	crf=CRF(name="crf1",ignore_last_label=True)
	result=crf(t)

	model=Model(input=[sen_in,mask_in,position_in],outputs=result)
	model.summary()
	return model,crf

def model2():#模型主体2
	sen_in=Input(shape=(k,r))
	mask_in=Input(shape=(k,r))
	position_in=Input(shape=(k,r))
	sen,mas,positi=sen_in,mask_in,position_in

	positi=Lambda(lambda x: x[0] * x[1])([positi, mas])
	#t=Add()([sen,positi])
	#t=Masking()(sen)
	#t=Dropout(0.25)(t)
	#t=Lambda(lambda x: x[0] * x[1])([t, mas])

	t=sen

	#t=Lambda(lambda x: x[0] * x[1])([t, mas])
	
	t=Bidirectional(CuDNNGRU(256,return_sequences=True,name="gru11"))(t)
	t=Dropout(0.25)(t)
	t=Lambda(lambda x: x[0] * x[1])([t, mas])
	
	t=LN()(t)

	t=Bidirectional(CuDNNGRU(256,return_sequences=True,name="gru21"))(t)
	t=Dropout(0.25)(t)
	t=Lambda(lambda x: x[0] * x[1])([t, mas])

	t=LN()(t)

	t=Bidirectional(CuDNNGRU(256,return_sequences=True,name="gru31"))(t)
	t=Dropout(0.25)(t)
	t=Lambda(lambda x: x[0] * x[1])([t, mas])

	t=LN()(t)


	t3=t
	t3=ONLSTM(512,4,return_sequences=True,name="on3")(t3)
	t3=Dropout(0.25)(t3)
	t3=Lambda(lambda x: x[0] * x[1])([t3, mas])
	
	
	t3=Dense(4,name="D3")(t3)
	t3=Dropout(0.25)(t3)
	result = t3
	









	crf=CRF(name="crf1",ignore_last_label=True)
	result=crf(result)

	model=Model(input=[sen_in,mask_in,position_in],outputs=result)
	model.summary()
	model_part_a=Model(input=[sen_in,mask_in,position_in],outputs=t)
	model_part_a.summary()
	model_part_b3=Model(input=[sen_in,mask_in,position_in],outputs=t3)
	model_part_b3.summary()
	return model,crf,model_part_a,model_part_b3

model,crf,model_part_a,model_part_b3=model2()

#回调器部分
from keras.callbacks import Callback
def data_generator(sentences,mask,tag,position,batch_size=Batch_size):#为每批次训练生成输入数据
	s=[]
	m=[]
	t=[]
	posi=[]
	while True:
		for i1 in range(len(sentences)):
			s.append(sentences[i1])
			m.append(mask[i1])
			t.append(tag[i1])
			posi.append(position[i1])
			if len(s)==batch_size:
				yield [np.array(s),np.array(m),np.array(posi)],[np.array(t)]
				s=[]
				m=[]
				t=[]
				posi=[]
	


class Evaluator(Callback):
	#每批次后调用此类，用于保存模型参数和评估
	def __init__(self):
		self.best_val_f1=0

	def on_epoch_end(self,epoch,logs=None):
		print("result:",model.predict_on_batch([vsentences[3:4,:,:],vmask[3:4,:],vposition[3:4,:,:]]))
		print("tag:",vtag[3:4,:,:])
		precision,recall,f1=evaluate(vsentences,vmask,vtag,vposition)#evaluate 为评估函数需要后面队友写 valid_data为测试数据集
		if f1>=self.best_val_f1:
			self.best_val_f1=f1
			model_part_b3.save_weights('best_model_model_part_b3.weights')
		print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
         )       

#main
adam=l_bei(Adam)
adam=adam(paramwise_lr_schedule={"gru11":0.00000001,"gru21":0.00000001,"gru31":0.00000001})
#adam=l_shiying(Adam)
#adam=adam()
#adam=l_fenduan(Adam)
#adam=adam(lr_schedule={40: 1,50:0.01})
model.compile(loss=crf.loss, optimizer=adam,metrics=[crf.accuracy])
model_part_a.load_weights('best_model_part1.weights')
evaluator = Evaluator()
model.fit_generator(
        data_generator(sentences,mask,tag,position),
        steps_per_epoch=int(len(sentences)/Batch_size),
        epochs=Epochs,callbacks=[evaluator])
