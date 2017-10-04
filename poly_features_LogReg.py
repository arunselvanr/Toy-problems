import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

import os
path = os.getcwd() + '/ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data.head()

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def gra_tanh(z):
    return (1 - np.power(tanh(z), 2))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def grad_sigmoid(z):
    return (np.exp(-z) / (1 + np.exp(-z)))

learning_rate = np.exp(-0)

x = np.array([[data.iloc[idx, 0] for idx in range(data.shape[0])],
              [data.iloc[idx, 1] for idx in range(data.shape[0])]])
x = np.transpose(x)

y = np.array([data.iloc[idx, 2] for idx in range(data.shape[0])])

def poly_features(xp):
    features = []
    for idx in range(data.shape[0]):
        features.append([1, xp[idx, 0], xp[idx, 1]])
        #features.append([1, xp[idx, 0]**2, xp[idx, 0] * xp[idx, 1], xp[idx, 1]**2 ])
        #features.append([1, xp[idx, 0]**3, np.power(xp[idx, 0], 2) * xp[idx, 1], np.power(xp[idx, 1], 2) * xp[idx, 0], xp[idx, 1]**3])

    return(np.array(features))

x_features = poly_features(x)
W = np.zeros((x_features.shape[1], 1))
a = 0.0
b = 0.0
for idx in range(x_features.shape[1]):
    W[idx, 0] = np.random.uniform(-.3, .3)



for step_size in range(1000):
    g = [0.0 for ix in range(x_features.shape[1])]
    for idx in range(data.shape[0]):
        a = np.dot(x_features[idx, :], W)
        b = sigmoid(a)

        if b == 0.0:
            for jdx in range(x_features.shape[1]):
                g[jdx] -= -((1 - y[idx]) / (1 - b)) * x_features[idx, jdx] * grad_sigmoid(a)

        elif b == 1.0:
            for jdx in range(x_features.shape[1]):
                g[jdx] -= (y[idx] / b) * x_features[idx, jdx] * grad_sigmoid(a)
        else:
            for jdx in range(x_features.shape[1]):
                g[jdx] -= ((y[idx] / b) - ((1 - y[idx]) / (1 - b))) * x_features[idx, jdx] * grad_sigmoid(a)

    #if [g[ix]**2 for ix in range(x_features.shape[1])] <= [np.exp(-16) for ix in range(x_features.shape[1])]:
     #   learning_rate = np.exp(3)
    #else:
     #   learning_rate = np.exp(-4)

    for feature in range(x_features.shape[1]):
        W[feature, 0] -= learning_rate * (g[feature] / data.shape[0])
    print(g)

#Predictions. The sigmoid is interpreted as probability
y_out = np.zeros((data.shape[0], 1))
for idx in range(data.shape[0]):
    if sigmoid(np.dot(x_features[idx, :], W)) >= .5:
        y_out[idx, 0] = 1
    else:
        y_out[idx, 0] = 0

x_pass1 = [x[idx, 0] for idx in range(data.shape[0]) if y[idx] == 1.0]
x_pass2 = [x[idx, 1] for idx in range(data.shape[0]) if y[idx] == 1.0]
plt.scatter(x_pass1, x_pass2, color = 'blue')
x_pass1 = [x[idx, 0] for idx in range(data.shape[0]) if y_out[idx] == 1.0]
x_pass2 = [x[idx, 1] for idx in range(data.shape[0]) if y_out[idx] == 1.0]
plt.scatter(x_pass1, x_pass2, color = 'green', marker = 'x')
plt.show()
x_fail1 = [x[idx, 0] for idx in range(data.shape[0]) if y[idx] == 0.0]
x_fail2 = [x[idx, 1] for idx in range(data.shape[0]) if y[idx] == 0.0]
plt.scatter(x_pass1, x_pass2, color = 'red')
x_fail1 = [x[idx, 0] for idx in range(data.shape[0]) if y_out[idx] == 0.0]
x_fail2 = [x[idx, 1] for idx in range(data.shape[0]) if y_out[idx] == 0.0]
plt.scatter(x_fail1, x_fail2, color = 'black', marker = 'x')
plt.show()

#print(np.transpose(y_out))
#Number of correct predictions.
correct  = 0
for idx in range(data.shape[0]):
    if y[idx] == y_out[idx]:
        correct += 1

print(correct/ float(data.shape[0]))