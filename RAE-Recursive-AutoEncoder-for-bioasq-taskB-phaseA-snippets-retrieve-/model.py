
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam


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






class autocoder():
    def __init__(self,word_size=300):
        self.model=self.build_model(word_size)
        """
        encoder_model,decoder_model,cata_model,rae_model
        0             1              2         3
        """
        
    def build_model(self,word_size):
        t_in=Input(shape=(None,1,word_size*2))
        tag_in=Input(shape=(None,2))
        n1_in=Input(shape=(None,1))
        n2_in=Input(shape=(None,1))
        t,tag,n1,n2=t_in,tag_in,n1_in,n2_in

        t=self.dilated_gated_conv1d(seq=t, dilation_rate=1)
        t=self.dilated_gated_conv1d(seq=t, dilation_rate=2)
        t=self.dilated_gated_conv1d(seq=t, dilation_rate=5)
        t=self.dilated_gated_conv1d(seq=t, dilation_rate=1)
        encoder=Dense(word_size)(t)
        encoder_model=Model([t_in],encoder)

        decoder=Dense(word_size*2)(encoder)
        dc=self.dilated_gated_conv1d(seq=decoder, dilation_rate=1)
        dc=self.dilated_gated_conv1d(seq=dc, dilation_rate=5)
        dc=self.dilated_gated_conv1d(seq=dc, dilation_rate=2)
        dc=self.dilated_gated_conv1d(seq=dc, dilation_rate=1)
        o1=Lambda(lambda x : x[:,:word_size])(dc)
        o2=Lambda(lambda x : x[:,word_size:])(dc)
        decoder_model=Model([t_in],[o1,o2])

        cata=self.dilated_gated_conv1d(seq=encoder, dilation_rate=1)
        cata=Lambda(lambda x : K.sum(x,axis=0))(cata)
        cata=Dense(2,'softmax')(cata)
        cata_model=Model([t_in,tag_in],cata)

        def mean_squared_error(y_true, y_pred):
            if not K.is_tensor(y_pred):
                y_pred = K.constant(y_pred)
            y_true = K.cast(y_true, y_pred.dtype)
            return K.mean(K.square(y_pred - y_true), axis=-1)
        o1_loss=(n1/(n1+n2))*(mean_squared_error(t1,o1))
        o2_loss=(n2/(n1+n2))*(mean_squared_error(t2,o2))
        c_loss=categorical_crossentropy(tag,cata)
        loss=(o1_loss+o2_loss)+(c_loss)

        rae_model=Model([t_in,tag_in,n1_in,n2_in],[o1,o2,cata])
        rae_model.add_loss(loss)
        rae_model.compile(optimizer=Adam(1e-3))
        rae_model.summary()

        return encoder_model,decoder_model,cata_model,rae_model
    



        



        












