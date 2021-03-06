import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten

from keras.optimizers import Adam



# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called

# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )

(X_train, y_train), (X_test, y_test) = mnist.load_data()



# data pre-processing

X_train = X_train.reshape(-1, 1,28, 28)/255.

X_test = X_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model=Sequential()
model.add(
    Convolution2D(
        nb_filter=32,#核数量
        
        nb_row=5,#核长宽
        nb_col=5,
        border_mode='same',
        input_shape=(1,28,28))#输入参数

        
    )

model.add(Activation('relu'))#y=af(wx) exp:tanh sigmoid relu 或者自己创造（必须可以微分） CNN relu   RNN tanh

#pool层 压缩长宽

model.add(MaxPooling2D(
    pool_size=(2,2),
                       strides=2,
                       border_mode='same', #padding method

                       )
          )

model.add(
    Convolution2D(64,5,5,border_mode='same')
    
    )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),
                       
                       border_mode='same', #padding method

                       )
          )
#全连接层
model.add(Flatten())#变成一dim
model.add(Dense(1024))
model.add(Activation('relu'))
#2全连接层
model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr=1e-4)

model.compile(optimizer=adam,

              loss='categorical_crossentropy',

              metrics=['accuracy'])

print('train')
model.fit(X_train,y_train,epochs=2,batch_size=64)

print('test')
loss,result=model.evaluate(X_test,y_test)

print(loss,result)

