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
#data.insert(0, 'Ones-again', 1.0)

#def tanh(z):
 #   return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

#def gra_tanh(z):
 #   return (1 - np.power(tanh(z), 2))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def grad_sigmoid(z):
    return (np.exp(-z) / float(np.power((1 + np.exp(-z)), 2)))

x = np.array([data.iloc[:, idx] for idx in range(data.shape[1] - 1)])
#x.shape = (3, 118)
y = np.transpose(np.array([data.iloc[:, -1]])) #y.shape = (118, 1)
label = range(2)
y_matrix = np.array([[int(label[label_count] == y[idx, 0])
                                   for idx in range(y.shape[0])] for label_count in range(2)])
y_out = np.zeros(y_matrix.shape)

#Tunable Parameters
lambdaa = 1.0 #Regularization constant (Don't know how it works)
hidden_size = 10 #Size of the hidden layer.
init_const = 1 #Used to initialize the weight matrices.
output_size = 2 #Size of the output layer.
###################################finding polynomial features###########################################
W1 = np.array([[np.random.normal(0.0, 1.0) * init_const for idx in range(x.shape[0])]
              for jdx in range(hidden_size)])
W2 = np.array([[np.random.normal(0.0, 1.0) * init_const for idx in range(hidden_size)]
              for jdx in range(output_size)])
W = [np.random.normal(0.0, 1.0) * init_const for idx in
     range(W1.shape[0]*W1.shape[1] + W2.shape[0]*W2.shape[1])]
#print(W1.shape) #(25, 401)
#print(W2.shape) #(10, 25)

def feed_forward(xf, yf, W1f, W2f):
    #xf.shape = (401, 1)
    A1f = np.dot(W1f, xf) #(25, 1)
    #print(A1f.shape)
    B1f = np.array([[sigmoid(A1f[ix, 0])] for ix in range(A1f.shape[0])]) #(25, 1)
    #print(B1f.shape)
    A2f = np.dot(W2f, B1f) #(10, 1)
    #print(A2f.shape)
    B2f = np.array([[sigmoid(A2f[ix, 0])] for ix in range(A2f.shape[0])]) #(10, 1)
    #print(B2f.shape)
    lossf = np.sum((yf * np.log(B2f)) +
                   ((1 - yf) * (1 - np.log(B2f))))
    return(lossf)

def loss(xl, yl, W1l, W2l, lambdal):
    lossl = 0.0

    for count in range(xl.shape[1]):
        lossl -= feed_forward(xl[:, count: (count + 1)], yl[:, count: (count + 1)], W1l, W2l)
        #print(lossl)
    lossl /= float(xl.shape[1])
    #Now for the regularizer term
    reg = 0.0
    #Sum up the squares of all elements of W1 except those from the first column
    for row_c in range(W1l.shape[0]):
        for col_c in range(W1l.shape[1] - 1):
            reg += np.power(W1l[row_c, col_c + 1], 2)
    #Now sum up all from W2
    for row_c in range(W2l.shape[0]):
        for col_c in range(W2l.shape[1]):
            reg += np.power(W2l[row_c, col_c], 2)

    reg *= lambdal
    reg /= float(2 * xl.shape[1])
    reg = 0.0
    return (lossl + reg)

def Loss_function(WL, xL, yL, W1L, W2L, lambdaL):
    W_count = 0
    for i in range(W1L.shape[0]):
        for j in range(W1L.shape[1]):
            W1L[i,j] = WL[W_count]
            W_count += 1

    for i in range(W2L.shape[0]):
        for j in range(W2L.shape[1]):
            W2L[i,j] = WL[W_count]
            W_count += 1

    return loss(xL, yL, W1L, W2L, lambdaL)

def BP(xb, yb, W1b, W2b, lambdab):
    #We require yg and xb to be a column vector
    A1b = np.dot(W1b, xb)  # (25, 1)
    B1b = np.array([[sigmoid(A1b[ix, 0])] for ix in range(A1b.shape[0])])  # (25, 1)
    A2b = np.dot(W2b, B1b)  # (10, 1)
    B2b = np.array([[sigmoid(A2b[ix, 0])] for ix in range(A2b.shape[0])])  # (10, 1)

    GA2 = B2b - yb
    GA1 = np.zeros((hidden_size, 1))
    #grad_A1 = np.array([[grad_sigmoid(A1b[ix, 0])] for ix in range(A1b.shape[0])])  # (25, 1)

    for ix in range(hidden_size):
        GA1[ix, 0] = np.dot(np.transpose(W2b[:, ix:(ix + 1)]), GA2) * grad_sigmoid(A1b[ix, 0])

    GW2 = np.dot(GA2, np.transpose(B1b)) #(10, 25)
    GW1 = np.dot(GA1, np.transpose(xb)) #(25, 401)

    return [GW1, GW2]

def gradient(Wg, xg, yg, W1g, W2g, lambdag):
    W_count = 0
    for i in range(W1g.shape[0]):
        for j in range(W1g.shape[1]):
            W1g[i, j] = Wg[W_count]
            W_count += 1

    for i in range(W2g.shape[0]):
        for j in range(W2g.shape[1]):
            W2g[i, j] = Wg[W_count]
            W_count += 1

    GW1g = np.zeros(W1g.shape)
    GW2g = np.zeros(W2g.shape)
    for count in range(xg.shape[1]):
        BP_call = BP(xg[:, count: (count + 1)], yg[:, count: (count + 1)], W1g, W2g, lambdag)
        GW1g += BP_call[0]
        GW2g += BP_call[1]
    #print(GW1g)
    GW1g = np.multiply(1 / float(xg.shape[1]), GW1g)
    #print(np.multiply(float(xg.shape[1]), GW1g))
    GW2g = np.multiply(1 / float(xg.shape[1]), GW2g)

    #Now for the regularization terms
    #GW1g += np.multiply((lambdag / float(xg.shape[1])), W1g)
    #GW1g[:, 0:1] -= np.multiply((lambdag / float(xg.shape[1])), W1g[:, 0:1])
    #GW2g += np.multiply(lambdag / float(xg.shape[1]), W2g)

    #Reassign the gradient vector
    GW = [0.0  for ix in range(len(Wg))]
    W_count = 0
    for i in range(GW1g.shape[0]):
        for j in range(GW1g.shape[1]):
            GW[W_count]  = GW1g[i, j]
            W_count += 1

    for i in range(GW2g.shape[0]):
        for j in range(GW2g.shape[1]):
            GW[W_count] = GW2g[i, j]
            W_count += 1
    #return [GW1g, GW2g]
    return GW #Use this if using a scipy gradient descent

##################Using scipy########################################

#result = opt.fmin_tnc(func=Loss_function, x0=W, fprime=gradient, args=(x, y_matrix, W1, W2, lambdaa))
#W = result[0]
#print(W)
fmin = opt.minimize(fun=Loss_function, x0=W, args=(x, y_matrix, W1, W2, lambdaa),
                method='TNC', jac=gradient)#, options={'maxiter': 10000})
print(fmin)
W = fmin.x
wcount = 0
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        W1[i, j] = W[wcount]
        wcount += 1

for i in range(W2.shape[0]):
    for j in range(W2.shape[1]):
        W2[i, j] = W[wcount]
        wcount += 1
###############handwritten GD###############
#learning_rate = np.exp(-2)
#for step_size in range(200):
 #   grad_call = gradient(W, x, y_matrix, W1, W2, lambdaa)
  #  W1 -= np.multiply(learning_rate, grad_call[0])
   # W2 -= np.multiply(learning_rate, grad_call[1])


for idx in range(x.shape[1]):
    A11 = np.dot(W1, x[:, idx: idx +1]) #(25, 1)
    B11 = np.array([[sigmoid(A11[ix, 0])] for ix in range(A11.shape[0])]) #(25, 1)
    A22 = np.dot(W2, B11) #(10, 1)
    B22 = np.array([[sigmoid(A22[ix, 0])] for ix in range(A22.shape[0])])
    y_out[:, idx: idx+1] = B22
    print(np.transpose(B22))

#for i in range(5):
    #print(np.transpose(y_out[:, 100+i: (101+ i)]))
    #print(np.transpose(y_matrix[:, 100+i: (101+ i)]))

for idx in range(y_out.shape[1]):
    maxx = max(y_out[:, idx])
    for jdx in range(y_out.shape[0]):
        if y_out[jdx, idx] == maxx:
            y_out[jdx, idx] = 1
        else:
            y_out[jdx, idx] = 0

correct = 0
for idx in range(y_out.shape[1]):
    if np.dot(np.transpose((y_out[:, idx: idx + 1] - y_matrix[:, idx: idx + 1])), (y_out[:, idx: idx + 1] - y_matrix[:, idx: idx + 1])) == 0.0:
        correct += 1

print(correct/ float(y_out.shape[1]))


#feed_forward(x[:, 0:1], y_matrix[:, 0:1], W1, W2)
#loss(x, y_matrix, W1, W2, lambdaa)
#Loss_function(W, x, y_matrix, W1, W2, lambdaa)
#BP(x[:, 0:1], y_matrix[:, 0:1], W1, W2, lambdaa)
gradient(W, x, y_matrix, W1, W2, lambdaa)

