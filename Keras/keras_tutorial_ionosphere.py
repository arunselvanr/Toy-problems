import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
dataset = np.loadtxt('ionosphere.data', delimiter=',')

X = dataset[:, 0:(dataset.shape[1] - 1)]
Y = dataset[: ,(dataset.shape[1] - 1)]
Y = np_utils.to_categorical(Y, 2)

model = Sequential()
model.add(Dense(500, activation='relu',input_dim=X.shape[1]))
model.add(Dense(500, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=20, batch_size=35, verbose=1)
score=model.evaluate(X,Y, verbose=1)
print 'Score = %f' %(score[1])