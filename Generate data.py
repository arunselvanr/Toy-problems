import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

a = 1.5
b = -1
sample_size = 250
x = [np.random.uniform(0, 5) for idx in range(sample_size)]
y = [np.random.uniform(0, 5) for idx in range(sample_size)]
z = []

for idx in range(sample_size):
    if x[idx] < a * y[idx] + b:
        z.append(1)
    else:
        z.append(0)


#plt.scatter(x, y, color = 'blue')
#plt.show()

red_x = [x[idx] for idx in range(sample_size) if z[idx] == 0]
red_y = [y[idx] for idx in range(sample_size) if z[idx] == 0]
plt.scatter(red_x, red_y, color = 'red')

blue_x = [x[idx] for idx in range(sample_size) if z[idx] == 1]
blue_y = [y[idx] for idx in range(sample_size) if z[idx] == 1]
plt.scatter(blue_x, blue_y, color = 'blue')
plt.show()


myfile = open("data2.txt", "a")

for idx in range(sample_size):
    myfile.write("%f,%f,%f\n" %(x[idx], y[idx], z[idx]))
    #myfile.write("\n")
    #print(x[idx], y[idx], z[idx], myfile)

myfile.close()
#myfile = open("test.txt", "r")
#content = myfile.readlines()
#content = [x.strip('\n') for x in content]
#myfile.close()

import os
path = os.getcwd() + '/test.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])



