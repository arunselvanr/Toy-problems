import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
#%matplotlib inline
from sklearn.preprocessing import OneHotEncoder
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
    return (sigmoid(z) * (1 - sigmoid(z)))

x = np.array([data.iloc[:, idx] for idx in range(data.shape[1] - 1)])
y = np.array([data.iloc[:, -1]])
label = range(2)
y_matrix = np.array([[int(label[label_count] == y[0, idx])
                                   for idx in range(y.shape[1])] for label_count in range(2)])
y_out = np.zeros(y_matrix.shape)
#Tunable Parameters
lambdaa = 1.0 #Regularization constant (Don't know how it works)
hidden_size = 4 #Size of the hidden layer.
init_const = 1.0 #Used to initialize the weight matrices.
output_size = 2 #Size of the output layer.

W1 = np.array([[np.random.uniform(-init_const, init_const) for i in range(x.shape[0])] for j in range(hidden_size)])
W2 = np.array([[np.random.uniform(-init_const, init_const) for i in range(hidden_size)] for j in range(output_size)])
#print(W1.shape) #(4, 3)
#print(W2.shape) #(2, 4)
wcount= 0
W = []
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        W.append(W1[i,j])
for i in range(W2.shape[0]):
    for j in range(W2.shape[1]):
        W.append(W2[i,j])

def Feed_Forward(x_f, y_f, W1_f, W2_f):
    A1_f = np.dot(W1_f, x_f)
    #print(A1_f)
    B1_f = np.array([[sigmoid(A1_f[ixf, 0])] for ixf in range(A1_f.shape[0])])
    #print(B1_f)
    A2_f = np.dot(W2_f, B1_f)
    #print(A2_f)
    B2_f = np.array([[sigmoid(A2_f[ixf, 0])] for ixf in range(A2_f.shape[0])])
    #print(B2_f)

    #Now to calculate the loss function
    #print(y_f * np.log(B2_f))
    #print((1 - y_f) * np.log(1 - B2_f))
    #print((y_f * np.log(B2_f)) + ((1 - y_f) * np.log(1 - B2_f)))
    loss_f = - np.sum((y_f * np.log(B2_f)) + ((1 - y_f) * np.log(1 - B2_f)))
    return loss_f

def Loss_function(W_l, x_l, y_l, W1_l, W2_l, lambdaa_l):
    wcountl = 0
    for il in range(W1_l.shape[0]):
        for jl in range(W1_l.shape[1]):
            W1_l[il, jl] = W_l[wcountl]
            wcountl += 1
    #print(W_l[wcountl:])
    for il in range(W2_l.shape[0]):
        for jl in range(W2_l.shape[1]):
            W2_l[il, jl] = W_l[wcountl]
            wcountl += 1
    #print(W2_l)
    loss_l = 0.0
    for run in range(x_l.shape[1]):
        #print(run)
        #print(y_l[:, run: run+1])
        #print(x_l[:, run: run + 1])
        loss_l += Feed_Forward(x_l[:, run: run + 1], y_l[:, run: run +1], W1_l, W2_l)
    #print(loss_l)
    loss_l /= float(x_l.shape[1])
    #print(loss_l * float(x_l.shape[1]))
    return loss_l

def Back_Prop(x_b, y_b, W1_b, W2_b):
    A1_b = np.dot(W1_b, x_b)
    gradA1_b = np.array([[grad_sigmoid(A1_b[ixb, 0])] for ixb in range(A1_b.shape[0])])
    B1_b = np.array([[sigmoid(A1_b[ixb, 0])] for ixb in range(A1_b.shape[0])])
    A2_b = np.dot(W2_b, B1_b)
    B2_b = np.array([[sigmoid(A2_b[ixb, 0])] for ixb in range(A2_b.shape[0])])

    GA2_b = B2_b - y_b #Partial derivative with respect to the inputs to the output layer.
    GB1_b = np.dot(np.transpose(W2_b), GA2_b) #Partial derivative with respect to the outputs of hidden layer.
    GA1_b = np.multiply(gradA1_b, GB1_b) #Partial derivative with respect to the inputs of hidden layer.
    #print(GA2_b)
    #print(W2_b)
    #print(np.transpose(W2_b))
    #print(GA2_b)
    #print(np.transpose(GB1_b))
    #print(gradA1_b)
    #print(GA1_b)
    #print(np.transpose(x_b))
    GW1_b = np.dot(GA1_b, np.transpose(x_b)) #Derivative wrt weight W1
    GW2_b = np.dot(GA2_b, np.transpose(B1_b)) #Derivative wrt weight W2
    #print(GW1_b)
    #print(1/ float(x.shape[1]))
    GW1_b = np.multiply(GW1_b, 1/ float(x.shape[1]))
    GW2_b = np.multiply(GW2_b, 1 / float(x.shape[1]))
    #print(GW1_b)
    return [GW1_b, GW2_b]

def gradient(W_g, x_g, y_g, W1_g, W2_g, lambdag):
    wcountg = 0
    for ig in range(W1_g.shape[0]):
        for jg in range(W1_g.shape[1]):
            W1_g[ig, jg] = W_g[wcountg]
            wcountg += 1
    for ig in range(W2_g.shape[0]):
        for jg in range(W2_g.shape[1]):
            W2_g[ig, jg] = W_g[wcountg]
            wcountg += 1

    GW1_g = np.zeros(W1_g.shape)
    GW2_g = np.zeros(W2_g.shape)
    for ixg in range(x_g.shape[1]):
        BP = Back_Prop(x_g[:, ixg: ixg +1], y_g[:, ixg: ixg +1], W1_g, W2_g)
        GW1_g += BP[0]
        GW2_g += BP[1]

    GW_g = []
    for ig in range(GW1_g.shape[0]):
        for jg in range(GW1_g.shape[1]):
            GW_g.append(GW1_g[ig, jg])
    for ig in range(GW2_g.shape[0]):
        for jg in range(GW2_g.shape[1]):
            GW_g.append(GW2_g[ig, jg])

    #print(GW1_g)
    #print(GW2_g)
    #print(GW_g[12:])
    return GW_g

#print(Feed_Forward(x[:, 0:1], y_matrix[:, 0:1], W1, W2))
#print(Loss_function(W, x, y_matrix, W1, W2, lambdaa))
#Back_Prop(x[:, 117:118], y_matrix[:, 117:118], W1, W2)
#gradient(W, x, y_matrix, W1, W2, lambdaa)

result = opt.fmin_tnc(func=Loss_function, x0=W, fprime=gradient, args=(x, y_matrix, W1, W2, lambdaa))
W = result[0]

wcount = 0
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        W1[i, j] = W[wcount]
        wcount += 1

for i in range(W2.shape[0]):
    for j in range(W2.shape[1]):
        W2[i, j] = W[wcount]
        wcount += 1

for idx in range(x.shape[1]):
    A11 = np.dot(W1, x[:, idx: idx +1]) #(25, 1)
    B11 = np.array([[sigmoid(A11[ix, 0])] for ix in range(A11.shape[0])]) #(25, 1)
    A22 = np.dot(W2, B11) #(10, 1)
    B22 = np.array([[sigmoid(A22[ix, 0])] for ix in range(A22.shape[0])])
    y_out[:, idx: idx+1] = B22
for i in range(100):
    print(np.transpose(y_out[:, 4+i: (5+ i)]))
    print(np.transpose(y_matrix[:, 4+i: (5+ i)]))


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

#for i in range(5):
 #   print(np.transpose(y_out[:, 4+i: (5+ i)]))
  #  print(np.transpose(y_matrix[:, 4+i: (5+ i)]))

print(correct/ float(y_out.shape[1]))

#Toy-problems
