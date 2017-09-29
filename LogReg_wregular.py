import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
#%matplotlib inline

import os
path = os.getcwd() + '/ex2data2.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
#data.insert(0, 'Ones', 1.0)
#data.insert(0, 'Ones-again', 1.0)

#def tanh(z):
 #   return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

#def gra_tanh(z):
 #   return (1 - np.power(tanh(z), 2))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def grad_sigmoid(z):
    return (np.exp(-z) / float(np.power((1 + np.exp(-z)), 2)))

#Tunable Parameters
lambdaa = 1.0 #Regularization constant (Don't know how it works)
#learning_rate = np.exp(-4) #When hand-coding GD
#Number_of_steps = 10000  #When hand-coding GD
degree = 2  #Degree of the polynomial feature
init_const = .1 #used to initialize the weight vector, Burak constant

x = np.array([data.iloc[:, idx] for idx in range(data.shape[1] - 1)]) #x.shape = (3, 118)
y = np.transpose(np.array([data.iloc[:, -1]])) #y.shape = (118, 1)
x_plot = x
def poly_features(xp, dp):
    features = []
    for idp in range(xp.shape[1]):
        feat = []
        feat.append(1.0)
        for deg in range(dp + 1):
            feat.append(np.power(xp[0, idp], dp - deg) * np.power(xp[1, idp], deg))
        features.append(feat)
    print(features)
    return(np.array(features))

x_features = poly_features(x, degree)
#print(x_features.shape)
x = np.transpose(x_features)
#print(x_features.shape)
W = np.array([np.random.normal(0, 1.0) * init_const for idx in range(x.shape[0])]) #W.shape = (3, ). This is needed to pass as parameter in scipy

def loss(Wll, xl, yll, lambdaal):
    Wl = np.array([Wll]) #Wl.shape = (1, 3)
    yl = np.transpose(yll)
    Al = np.dot(Wl, xl) #(1, 118)

    Bl = np.transpose(np.array([[sigmoid(Al[0, ixg])] for ixg in range(data.shape[0])])) #(1, 118)

    Loss_vector = (yl * np.log(Bl)) + ((1.0 - yl) * np.log(1.0 - Bl)) # (1, 118)

    loss = - np.sum(Loss_vector) / float(data.shape[0])
    #print(loss)
    #Time to introduce some regularization
    #print(np.sum((Wl * Wl)[0, 1:]))
    reg = lambdaal * np.sum((Wl * Wl)[0, 1:]) / float(data.shape[0] * 2)
    #print(reg * float(data.shape[0] * 2))
    #reg = 0.0
    return(loss + reg) #We return the regularized loss funciton value

def gradient(Wgg, xg, ygg, lambdaag):
    Wg = np.array([Wgg])  # Wl.shape = (1, 3)
    yg = np.transpose(ygg)

    Ag = np.dot(Wg, xg)  # (1, 118)

    Bg = np.transpose(np.array([[sigmoid(Ag[0, ixg])] for ixg in range(data.shape[0])]))  # (1, 118)
    #print(Bg)
    Grad_vector1 = (Bg - yg) # (1, 118)
    #print(Grad_vector1)
    #Bgg = np.transpose(np.array([[grad_sigmoid(Ag[0, ixg])] for ixg in range(data.shape[0])])) # (1, 118)
    #print(Bgg)

    Grad_vector2 = np.transpose(np.array([[np.sum(Grad_vector1 * xg[ixg, :])] for ixg in range(xg.shape[0])])) # (1, 3)
    #print(Grad_vector2)
    #Now to calculate the derivative wrt the regularization part.

    Grad_vector = (Grad_vector2 + np.multiply(lambdaag, Wg))/ float(data.shape[0]) #The second term is from the regularization.
    #print('%f %f'%(Grad_vector[0,0], lambdaag * Wg[0,0] / float(data.shape[0])))
    Grad_vector[0,0] -= lambdaag * Wg[0,0] / float(data.shape[0])
    #print(Grad_vector)
    #print(Grad_vector[0,0])
    return([Grad_vector[0, ix] for ix in range(Grad_vector.shape[1])])


##############################################################SCIPY#############################################
#########################################################SCIPY##################################################
####################################################SCIPY#######################################################
result = opt.fmin_tnc(func=loss, x0=W, fprime=gradient, args=(x, y, lambdaa))

#print(result)
#print(W)
#print(x[:, 1])
#print(np.dot(W, x[:, 1]))

##############################################################SCIPY#############################################
#########################################################SCIPY##################################################
####################################################SCIPY#######################################################
#for step_size in range(Number_of_steps):
 #   gradient_call = gradient(W, x, y, lambdaa)
  #  W -= np.multiply(learning_rate, gradient_call)
   # print([gradient_call, W])
##################################Above is a simple gradient descent routine####################################
##################################Above is a simple gradient descent routine####################################
##################################Above is a simple gradient descent routine####################################

#W = result[0]



#print(gradient(W, x, y, lambdaa))
#loss(W, x, y, lambdaa)
################################################################################################################
################################Now we check how well things work##############################################
################################################################################################################
W = result[0]
y_out = np.zeros(y.shape)
for idx in range(data.shape[0]):
    if sigmoid(np.dot(W, x[:, idx])) >= .5:
        y_out[idx, 0] = 1.0
    else:
        y_out[idx, 0] = 0.0

correct = 0.0
for idx in range(data.shape[0]):
    if y[idx, 0] == y_out[idx, 0]:
        correct += 1.0

print(correct / data.shape[0])
print(np.transpose(y_out))

x_pass1 = [x_plot[0, idx] for idx in range(data.shape[0]) if y[idx, 0] == 1.0]
x_pass2 = [x_plot[1, idx] for idx in range(data.shape[0]) if y[idx, 0] == 1.0]
plt.scatter(x_pass1, x_pass2, color = 'blue')
x_pass1 = [x_plot[0, idx] for idx in range(data.shape[0]) if y_out[idx, 0] == 1.0]
x_pass2 = [x_plot[1, idx] for idx in range(data.shape[0]) if y_out[idx, 0] == 1.0]
plt.scatter(x_pass1, x_pass2, color = 'green', marker = 'x')
#plt.show()
x_fail1 = [x_plot[0, idx] for idx in range(data.shape[0]) if y[idx, 0] == 0.0]
x_fail2 = [x_plot[1, idx] for idx in range(data.shape[0]) if y[idx, 0] == 0.0]
plt.scatter(x_fail1, x_fail2, color = 'red')
x_fail1 = [x_plot[0, idx] for idx in range(data.shape[0]) if y_out[idx, 0] == 0.0]
x_fail2 = [x_plot[1, idx] for idx in range(data.shape[0]) if y_out[idx, 0] == 0.0]
plt.scatter(x_fail1, x_fail2, color = 'black', marker = 'x')
#plt.show()

