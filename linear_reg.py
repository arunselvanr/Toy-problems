import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

path = os.getcwd()+'/ex1data.txt'
data = pd.read_csv(path, header=None, names = ['Population', 'Profit'])

data.plot(kind = 'scatter', x = 'Population', y = 'Profit')
#plt.show()

data.insert(0, 'Ones', 1)

x = np.array(data.iloc[:, 1])
y = np.array(data.iloc[:, 2])

M = np.array([[1 for idx in range(data.shape[0])], x], dtype = np.float64)

theta = np.zeros((1,2), dtype = np.float64)

#Simple Gradient Descent
def error_gradient(y_f, error_f):
    grad1 = 0.0
    for i_f in range(data.shape[0]):
        grad1 += (y_f[0, i_f] - y[i_f])
    grad2 = 0.0
    for i_f in range(data.shape[0]):
        grad2 += (y_f[0, i_f] - y[i_f]) * x[i_f]

    return(np.array([grad1/ error_f, grad2/ error_f]))

constant_step_size = np.exp(-5)
for step_size in range(1000000):
    y_hat = np.dot(theta, M)
    error = 0.0
    for idx in range(data.shape[0]):
        error += (y[idx] - y_hat[0, idx]) ** 2
    correction = error_gradient(y_hat, error)
    theta[0, 0] -= constant_step_size * correction[0]
    theta[0, 1] -= constant_step_size * correction[1]

y_hat = np.dot(theta, M)

plt.scatter(x, y_hat,color='green')
plt.show()