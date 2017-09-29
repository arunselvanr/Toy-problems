import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
#%matplotlib inline

data = loadmat('ex3data1.mat')
x = data["X"]
Y = data['y']
label = range(10)
label[0] = 10
y_matrix = np.transpose(np.array([[1 - int(label[label_count] ==  Y[idx, 0]) for idx in range(Y.shape[0])] for label_count in range(10)]))
y_out = np.zeros(y_matrix.shape)
#y_matrix.shape = (5000, 10)
#print(y_matrix.shape)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def grad_sigmoid(z):
    return (np.exp(-z) / float(np.power((1 + np.exp(-z)), 2)))

#Tunable Parameters
lambdaa = 1.0 #Regularization constant (Don't know how it works)
#learning_rate = np.exp(-4) #When hand-coding GD
#Number_of_steps = 10000  #When hand-coding GD
degree = 1  #Degree of the polynomial feature
init_const = .1 #used to initialize the weight vector, Burak constant
###################################finding polynomial features###########################################
x_append = np.insert(x, 0, 1.0, axis = 1) # (5000, 401)
x = np.transpose(x_append) # (401, 5000)
for lab in range(10):
    W = np.array([np.random.normal(0, 1.0) * init_const for idx in
                  range(x.shape[0])])  # W.shape = (401, ). This is needed to pass as parameter in scipy
    y = y_matrix[:, lab:(lab +1)] # (5000, 1)

    def loss(Wll, xl, yll, lambdaal):
        Wl = np.array([Wll]) #Wl.shape = (1, 401)
        yl = np.transpose(yll) # (1, 5000)
        Al = np.dot(Wl, xl) #(1, 118)

        Bl = np.transpose(np.array([[sigmoid(Al[0, ixg])] for ixg in range(x_append.shape[0])])) #(1, 118)

        Loss_vector = (yl * np.log(Bl)) + ((1.0 - yl) * np.log(1.0 - Bl)) # (1, 118)

        loss = - np.sum(Loss_vector) / float(x_append.shape[0])
        #print(loss)
        #Time to introduce some regularization
        #print(np.sum((Wl * Wl)[0, 1:]))
        reg = lambdaal * np.sum((Wl * Wl)[0, 1:]) / float(x_append.shape[0] * 2)
        #print(reg * float(data.shape[0] * 2))
        #reg = 0.0
        return(loss + reg) #We return the regularized loss funciton value

    def gradient(Wgg, xg, ygg, lambdaag):
        Wg = np.array([Wgg])  # Wl.shape = (1, 3)
        yg = np.transpose(ygg)

        Ag = np.dot(Wg, xg)  # (1, 118)

        Bg = np.transpose(np.array([[sigmoid(Ag[0, ixg])] for ixg in range(x_append.shape[0])]))  # (1, 118)
        #print(Bg)
        Grad_vector1 = (Bg - yg) # (1, 118)
        #print(Grad_vector1)
        #Bgg = np.transpose(np.array([[grad_sigmoid(Ag[0, ixg])] for ixg in range(data.shape[0])])) # (1, 118)
        #print(Bgg)

        Grad_vector2 = np.transpose(np.array([[np.sum(Grad_vector1 * xg[ixg, :])] for ixg in range(xg.shape[0])])) # (1, 3)
        #print(Grad_vector2)
        #Now to calculate the derivative wrt the regularization part.

        Grad_vector = (Grad_vector2 + np.multiply(lambdaag, Wg))/ float(x_append.shape[0]) #The second term is from the regularization.
        #print('%f %f'%(Grad_vector[0,0], lambdaag * Wg[0,0] / float(data.shape[0])))
        Grad_vector[0,0] -= lambdaag * Wg[0,0] / float(x_append.shape[0])
        #print(Grad_vector)
        #print(Grad_vector[0,0])
        return([Grad_vector[0, ix] for ix in range(Grad_vector.shape[1])])


##############################################################SCIPY#############################################
#########################################################SCIPY##################################################
####################################################SCIPY#######################################################
    result = opt.fmin_tnc(func=loss, x0=W, fprime=gradient, args=(x, y, lambdaa))
#gradient(W, x, y, lambdaa)
    W = np.array(result[0])
    for idx in range(x_append.shape[0]):
    #print(sigmoid(np.dot(W, x[:, idx])))
        if sigmoid(np.dot(W, x[:, idx])) >= .5:
            y_out[idx, lab] = 1.0
        else:
            y_out[idx, lab] = 0.0

    correct = 0.0
    for idx in range(x_append.shape[0]):
        if y[idx, 0] == y_out[idx, lab]:
            correct += 1.0

    print(correct/float(x_append.shape[0]))