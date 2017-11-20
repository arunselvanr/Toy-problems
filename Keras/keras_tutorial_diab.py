import numpy as np
np.random.seed(1234)
import random

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

dataset = np.loadtxt('pima-indians-diabetes.data', delimiter=',')

X= dataset[:, 0:8]
Y = dataset[:, 8]
Y = np_utils.to_categorical(Y, 2)
#####MODEL  BUILDING#########
model = Sequential()
model.add(Dense(300, activation='relu', input_dim=X.shape[1]))
model.add(Dense(300, activation='relu'))
model.add(Dropout(.4))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X, Y, epochs=2000, batch_size=48, verbose=1)
print('model fitting is done')
score = model.evaluate(X , Y, verbose=1)
print(score)
