import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam
TIME_STEPS = 1     # same as the height of the image
INPUT_SIZE = 400    # same as the width of the image
BATCH_INDEX = 0
OUTPUT_SIZE = 2
BATCH_SIZE=5000
CELL_SIZE =5000
LR = 0.001
print('loads.....')
X_train=np.loadtxt('tvec.txt',dtype=np.float64)
testp_vec=np.loadtxt('epvec.txt',dtype=np.float64)
testo_vec=np.loadtxt('eovec.txt',dtype=np.float64)
X_test=np.concatenate((testp_vec,testo_vec),axis = 0)
p_tag = np.ones(31714)
n_tag = np.zeros(31060)
y_train=np.hstack((n_tag,p_tag))
test_p_tag=np.ones(9323)
test_n_tag=np.zeros(10431)
y_test=np.hstack((test_p_tag,test_n_tag))
X_train = X_train.reshape(-1,1 ,400)      
X_test = X_test.reshape(-1, 1, 400)
y_test=y_test.reshape(19754,1)
y_train=y_train.reshape(62774,1)

y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)
print(y_test)
print('........load succe')

#build RNN model
model = Sequential()
#RNN cell
model.add(SimpleRNN(batch_input_shape=(None,TIME_STEPS,INPUT_SIZE),output_dim=CELL_SIZE,))
#output layer
model.add(Dense(OUTPUT_SIZE))   
model.add(Activation('softmax'))
#optimizer
adam = Adam(LR)
model.compile(optimizer=adam,loss='mean_squared_erro',metrics=['accuracy'])
# training
cost = model.fit(X_train, y_train)
cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
print('test cost: ', cost, 'test accuracy: ', accuracy)


        
           
