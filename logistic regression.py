import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
#%matplotlib inline

import os
path = os.getcwd() + '/ex2data2.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data.insert(0, 'Ones', 1.0)

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def gra_tanh(z):
    return (1 - np.power(tanh(z), 2))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def grad_sigmoid(z):
    return np.exp(-z) / (1 + np.exp(-z))

x = np.array([data.iloc[: , 0 ], data.iloc[:, 1], data.iloc[:, 2]])
print(x.shape)
#for idx in range(data.shape[0]):
 #   x[idx, 0] /= max(max_x0, max_x1)
  #  x[idx, 1] /= max(max_x0, max_x1)

y = np.array([data.iloc[idx, 2] for idx in range(data.shape[0])])
#print(y.shape)
W = [np.random.normal(0.0, 1.0)*.01 for idx in range(x.shape[0])]

def loss(Wll, xl, yl):
    Wl = np.array([Wll])
    #print(xl.shape)
    Al = np.dot(Wl, xl)
    #print(yl.shape)
    Bl = np.array([sigmoid(Al[0, ixl])  for ixl in range(Al.shape[1])])
    one_minus_yl = np.full(yl.shape, 1.0) - yl
    #print(np.full(Bl.shape, 1.0))
    one_minus_Bl = np.full(Bl.shape, 1.0) - Bl
    return(- np.sum(y * np.log(Bl)) / data.shape[0] - np.sum(one_minus_yl * np.log(one_minus_Bl)) / data.shape[0])

def gradient(Wgg, xg, yg):
    Wg = np.array([Wgg])
    # print(xg.shape)
    Ag = np.dot(Wg, xg)
    # print(yg.shape)
    Bg = np.array([sigmoid(Ag[0, ixg]) for ixg in range(Ag.shape[1])])
    grad_Ag = np.array([grad_sigmoid(Ag[0, ixg]) for ixg in range(Ag.shape[1])])
    one_minus_yg = np.full(yg.shape, 1.0) - yg
    one_minus_Bg = np.full(Bg.shape, 1.0) - Bg
    #print(np.sum((yg / Bg - one_minus_yg / one_minus_Bg).shape == grad_Ag.shape)
    grad = [np.sum((yg / Bg - one_minus_yg / one_minus_Bg) * grad_Ag * np.full(grad_Ag.shape, xg[ixg]))/float(data.shape[0]) for ixg in range(Wg.shape[1])]
    return(grad)

result = opt.fmin(func=loss, x0=W, fprime = gradient, args=(x, y))

W = np.array(result[0])
print(W)
y_out = np.zeros((data.shape[0], 1))
for idx in range(data.shape[0]):
    #print(sigmoid(np.dot(W, x[:, idx])))
    if sigmoid(np.dot(W, x[:, idx])) >= .5:
        y_out[idx, 0] = 1.0
    else:
        y_out[idx, 0] = 0.0

correct = 0.0
for idx in range(data.shape[0]):
    if y[idx] == y_out[idx]:
        correct += 1.0

print(correct/float(data.shape[0]))