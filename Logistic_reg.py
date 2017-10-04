import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

path = os.getcwd()+'/ex2data.txt'

data = pd.read_csv(path, header=None, names = ['Exam1', 'Exam2', 'Result'])

exam1_pass = [data.iloc[idx, 0] for idx in range(data.shape[0]) if data.iloc[idx, 2] == 1]
exam2_pass =  [data.iloc[idx, 1] for idx in range(data.shape[0]) if data.iloc[idx, 2] == 1]
#plt.scatter(exam1_pass, exam2_pass, color = 'green', marker = 'o')

exam1_fail = [data.iloc[idx, 0] for idx in range(data.shape[0]) if data.iloc[idx, 2] == 0]
exam2_fail =  [data.iloc[idx, 1] for idx in range(data.shape[0]) if data.iloc[idx, 2] == 0]
#plt.scatter(exam1_fail, exam2_fail, color = 'red', marker = 'x')
#plt.show()

def sigmoid(z):
    return float(1/(1 + np.exp(-z)))

def grad_sigmoid(z):
    return float(( np.exp(z)/ np.power(1 + np.exp(-z), 2)))

def tanh(z):
    return float((np.exp(z) - np.exp(- z))/ (np.exp(z) + np.exp(- z)))

def grad_tanh(z):
    return float(1 - np.power (tanh(z), 2))

########################################################################Plot the sigmoid
#x_plot = np.arange(-10, +10, .1)
#y_plot = []
#for idx in x_plot:
#    y_plot.append(sigmoid(idx))
#plt.plot(x_plot, y_plot)
#plt.show()
########################################################################Plot the sigmoid

ip_dim = 2
x = np.array([[float(data.iloc[idx,0]) for idx in range(data.shape[0])], [data.iloc[jdx,1] for jdx in range(data.shape[0])]])
x = np.transpose(x)
y = [float(data.iloc[idx, 2]) for idx in range(data.shape[0])]
W_h = np.zeros((ip_dim,ip_dim))
W_k = np.zeros((ip_dim,1))

def BP(W_hb, W_kb, b_kb, a_kb, b_hb, a_hb, yb):
    if b_kb[0,0] == 0.0:
        g_0 = 1 - yb
    elif b_kb[0,0] == 1.0:
        g_0 = - yb
    else:
        g_0 = - yb / b_kb[0,0] + (1-yb) / (1 - b_kb[0,0]) # grad wrt b_k which is also y_out
    #print(g_0)
    g_1  = g_0 * grad_sigmoid(a_kb[0,0]) # grad wrt a_k
    #print(g_1)
    GW_kb = np.zeros((ip_dim, 1)) # grad wrt the weight matrix in stage 1 (backwards)
    GW_kb[0,0] = b_hb[0,0] * g_1
    GW_kb[1,0] = b_hb[1,0] * g_1
    #print(GW_k)
    g_2 = np.zeros((ip_dim,1)) # grad wrt b_h, o/p of the hidden layer
    g_2[0, 0] = g_1 * W_kb[0, 0]
    g_2[1, 0] = g_1 * W_kb[1, 0]

    g_3 = np.zeros((ip_dim, 1))  # grad wrt a_h, i/p to the hidden layer
    g_3[0, 0] = g_2[0,0] * grad_tanh(a_hb[0,0])
    g_3[1, 0] = g_2[1,0] * grad_tanh(a_hb[1,0])
    GW_hb = np.array([[g_3[0, 0] * x[0, 0], g_3[0, 0] * x[1, 0]], [g_3[1, 0] * x[0, 0], g_3[1, 0] * x[1, 0]]])# grad wrt the weight matrix in stage 2 (backwards)
    #print(GW_hb)
    return([GW_hb, GW_kb])

def loss(W_hl, W_kl, xl, yl):
    a_hl = np.dot(W_hl, xl)
    b_hl = a_hl
    for idx in range(ip_dim):
        b_hl[idx, 0] = tanh(a_hl[idx, 0])
    a_kl = np.dot(np.transpose(W_kl), b_hl)
    b_kl = a_kl
    b_kl[0, 0] = sigmoid(a_kl[0, 0])
    #print(b_k[0,0])
    BP_call = BP(W_hl, W_kl, b_kl, a_kl, b_hl, a_hl, yl)
    return BP_call

def Grad_loss(W_hg, W_kg, xg, yg):
    GW_kg = np.zeros((ip_dim, 1))
    GW_hg = np.zeros((ip_dim,ip_dim))
    #loss_value = 0.0
    for idx in range(data.shape[0]):
        x_colg = np.array([[xg[idx, 0]], [xg[idx, 1]]])
        Grad_data = loss(W_hg, W_kg, x_colg, yg[idx])
        #loss_value += np.power(Grad_data[0], 2)
        #print(Grad_data[1])
        GW_hg += Grad_data[0]
        GW_kg += Grad_data[1]
        #print(GW_hg)
        #print(GW_kg)
    #loss_value = np.sqrt(loss_value)
    #loss_M = np.full(GW_h.shape, loss_value)
    #GW_h /= loss_M
    #loss_M1 = np.full(GW_k.shape, loss_value)
    #GW_k /= loss_M1
    return([GW_hg, GW_kg])

########################################################################################################################
########################################################################################################################
#######################################Going for gradient descent now###################################################
########################################################################################################################
########################################################################################################################
W_h = np.full((ip_dim,ip_dim), 0.5)
W_k = np.full((ip_dim, 1), 0.5)
constant_step_size = np.exp(-8)
css_Mh = np.full(W_h.shape, constant_step_size)
css_Mk = np.full(W_k.shape, constant_step_size)

for step_size in range(5000):
    grad_loss_call = Grad_loss(W_h, W_k, x, y)
    W_h -= css_Mh * grad_loss_call[0]
    W_k -= css_Mk * grad_loss_call[1]
    print(W_k)
    #print(W_h)

################ Some analytics #####################
#print(grad_tanh(-2))
#print(W_h)
#print(W_k)