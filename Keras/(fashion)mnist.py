#This is written with theano as the backend
import keras
import numpy as np
np.random.seed(1234)

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Dense, Flatten

from keras.datasets import fashion_mnist
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#print(x_train.shape)
#from matplotlib import pyplot as plt
#plt.imshow(x_train[67])
#plt.show()
#print(max(y_train))

#Preprocessing the data for using keras with theano as the backend
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])
x_train /= 255 #This is merely rescaling for efficiency.
x_test /= 255  #As above.
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#Let start building the model
model = Sequential()
model.add(Convolution2D(50, (3, 3), activation='relu', data_format='channels_first', input_shape=(1, 28, 28)))
model.add(Convolution2D(50, (3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.25))
model.add(Flatten())
model.add(Dense(40, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(10, activation='softmax'))
#model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=6, verbose=1)
score = model.evaluate(x_test, y_test, verbose=1)

print(score)
