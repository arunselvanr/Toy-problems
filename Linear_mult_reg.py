import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

path = os.getcwd()+'/ex1data2.txt'
data = pd.read_csv(path, header=None, names = ['Size', 'Bedroom', 'Price'])
print(min(data.iloc[:, 2]))
print(max(data.iloc[:, 2]))
print(np.median(data.iloc[:, 2]))

size_low = [data.iloc[idx, 0] for idx in range(data.shape[0]) if data.iloc[idx, 2] <= np.median(data.iloc[:, 2])]
bedroom_low  = [data.iloc[idx, 1] for idx in range(data.shape[0]) if data.iloc[idx, 2] <= np.median(data.iloc[:, 2])]
#plt.scatter(size_low, bedroom_low, color = 'red', marker = 'o')


size_high = [data.iloc[idx, 0] for idx in range(data.shape[0]) if data.iloc[idx, 2] > np.median(data.iloc[:, 2])]
bedroom_high  = [data.iloc[idx, 1] for idx in range(data.shape[0]) if data.iloc[idx, 2] > np.median(data.iloc[:, 2])]
#plt.scatter(size_high, bedroom_high, color = 'green', marker = 'x')
#plt.show()

x1 = [data.iloc[idx, 0] for idx in range(data.shape[0])]
x2 = [data.iloc[idx, 1] for idx in range(data.shape[0])]
y = [data.iloc[idx, 2] for idx in range(data.shape[0])]

M = np.array([x1, x2, [1 for idx in range(data.shape[0])]])

def error_gradient(y_hat_f, error_f):
    grad1 = 0.0
    grad2 = 0.0
    grad3 = 0.0
    for idx in range(data.shape[0]):
        grad1 += (y_hat_f[0, idx] - y[idx]) * x1[idx]
        grad2 += (y_hat_f[0, idx] - y[idx]) * x2[idx]
        grad3 += (y_hat_f[0, idx] - y[idx])
    error_f =  error_f
    return([ grad1/ error_f, grad2/ error_f, grad3/ error_f])

#Gradient descent
theta = np.zeros((1,3))
const_step_size = np.exp(-6)
for step_size in range(1000000):
    y_hat = np.dot(theta , M)
    error = 0.0
    for idx in range(data.shape[0]):
        error += (y[idx] - y_hat[0, idx])**2
    error = np.sqrt(error)
    if step_size % 100 == 0:
        print(error)
    correction = error_gradient(y_hat, error)

    theta[0, 0] -= const_step_size * correction[0]
    theta[0, 1] -= const_step_size * correction[1]
    theta[0, 2] -= const_step_size * correction[2]

y_hat = np.dot(theta, M)

#plt.scatter(x1, y , color = 'blue')
#plt.scatter(x1, y_hat, color = 'red')
#plt.show()
#plt.scatter(x2, y, color = 'blue' )
#plt.scatter(x2, y_hat, color = 'red')
#plt.show()

ax.scatter(x1, x2, y, zdir= 'z', color='red')
ax.scatter(x1, x2, y_hat, zdir= 'z')
plt.show()





