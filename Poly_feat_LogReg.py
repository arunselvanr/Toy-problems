import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
#%matplotlib inline

import os
path = os.getcwd() + '/ex2data2.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data.head()

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def gra_tanh(z):
    return (1 - np.power(tanh(z), 2))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def grad_sigmoid(z):
    return (np.exp(-z) / np.power((1 + np.exp(-z)), 2))

#Tunable Parameters
lambdaa = 1.0
learning_rate = np.exp(-4)

x = np.array([[data.iloc[idx, 0] for idx in range(data.shape[0])],
              [data.iloc[idx, 1] for idx in range(data.shape[0])]])
x = np.transpose(x)

y = np.array([[data.iloc[idx, 2]] for idx in range(data.shape[0])])

def poly_features(xp):
    features = []
    for idx in range(data.shape[0]):
        features.append([1, xp[idx, 0], xp[idx, 1]])
        #features.append([1, 1, xp[idx, 0]**2, xp[idx, 0] * xp[idx, 1], xp[idx, 1]**2 ])
        #features.append([1, 1, xp[idx, 0]**3, np.power(xp[idx, 0], 2) * xp[idx, 1], np.power(xp[idx, 1], 2) * xp[idx, 0], xp[idx, 1]**3])

    return(np.array(features))
x_features = poly_features(x)

#print(x_features.shape)
W = np.array([np.random.normal(0.0, 1.0) * .01  for idx in range(x_features.shape[1])])
#print(W.shape)
def loss(Wll, xl, yl, lambdal):
    Wl = np.array([[Wll[idxl]]  for idxl in range(x_features.shape[1])])
    Al = np.dot(xl, Wl)

    Bl = np.array([[sigmoid(Al[ixl, 0])] for ixl in range(Al.shape[0])])
    #print(Bl.shape)
    one_minus_yl = np.array([[1 - yl[ixl, 0]] for ixl in range(data.shape[0])])
    one_minus_Bl = np.array([[1 - Bl[ixl, 0]] for ixl in range(data.shape[0])])
    #print(one_minus_Bl)
    #print(np.multiply(one_minus_yl, np.log(one_minus_Bl)))
    #print(one_minus_Bl + one_minus_yl)

    Lossl = (y * np.log(Bl)) + (one_minus_yl * np.log(one_minus_Bl)) #Loss vector
    lossl = - np.sum(Lossl) / float(data.shape[0]) #Loss without the regularization term
    #print(lossl)

    #Now we calculate the regularization term

    regl = lambdal * np.sum((Wl * Wl)[1:,:]) / float(2 * data.shape[0])
    #print(regl + lossl)
    #Loss with the regularizer is returned as the loss function
    return(lossl + regl)

def grad(Wgg, xg, yg, lambdag):
    Wg = np.array([[Wgg[idxg]]  for idxg in range(x_features.shape[1])])
    Ag = np.dot(xg, Wg)
    #print(Ag.shape)
    Grad_Ag = np.array([[grad_sigmoid(Ag[ixg, 0])] for ixg in range(Ag.shape[0])])
    Bg = np.array([[sigmoid(Ag[ixg, 0])] for ixg in range(Ag.shape[0])])
    #print(Bg)
    one_minus_yg = np.array([[1 - yg[ixg, 0]] for ixg in range(data.shape[0])])
    one_minus_Bg = np.array([[1 - Bg[ixg, 0]] for ixg in range(data.shape[0])])
    #print(one_minus_Bg)
    Grad_outg = ( yg / Bg ) - (one_minus_yg / one_minus_Bg)
    Grad_hidg = Grad_outg * Grad_Ag
    Grad_wg = np.zeros(Wg.shape)
    for ixg in range(Wg.shape[0]):
        Grad_wg[ixg] = np.sum(Grad_hidg * xg[:, ixg:(ixg + 1)]) / float(data.shape[0])
    #print(Grad_wg)
    #print(yg /  Bg)
    #print(one_minus_yg / one_minus_Bg)
    #print(Lossg)

    Grad_regg = (np.full(Wg.shape, lambdag) * Wg) / np.full(Wg.shape, float(data.shape[0]))
    #print(Grad_regg)
    Grad_term = Grad_regg + Grad_wg
    Grad_term[0,0] -= Grad_regg[0, 0]/ float(data.shape[])
    return (Grad_term[:,0])

#grad(W, x_features, y, lambdaa)
#loss(W, x_features, y, lambdaa)
#result = opt.fmin_tnc(func=loss, x0=W, fprime=grad, args=(x_features, y, lambdaa))
#print(result)
#print(grad(W, x_features, y, lambdaa).shape)
const_ss = learning_rate
for step_size in range(6000):
    W -= (np.full(W.shape, const_ss) * grad(W, x_features, y, lambdaa))

#print(x_features[0,:].shape)
y_out = np.zeros((data.shape[0], 1))
for idx in range(data.shape[0]):
    if sigmoid(np.dot(W, x_features[idx, :])) >= .5:
        y_out[idx, 0] = 1.0
    else:
        y_out[idx, 0] = 0.0

correct = 0.0
for idx in range(data.shape[0]):
    if y[idx, 0] == y_out[idx, 0]:
        correct += 1.0
#print(y_out)
print(correct/float(data.shape[0]))
