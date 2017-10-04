from random import randint
from math import sqrt
import matplotlib.pyplot as plt
dim = 2
sample_size = 1000
k=10
x = []
for idx in range(sample_size):
 x.append([randint(-6000,6000) for jdx in range(dim)])

means=[]
for idx in range(k):
 means.append([randint(-6000,6000) for jdx in range(dim)])

def distance(x, y):
    norm = 0
    for idx in range(dim):
        norm += (x[idx] - y[idx])**2

    return(sqrt(norm))

def vector_sum(x, y):
    v_sum = []
    for idx in range(dim):
        v_sum.append(x[idx] + y[idx])
    return(v_sum)

cluster_no=[-1.0 for idx in range(sample_size)]
dist = [ -1.0 for idx in range(sample_size)]
while True:
    change = 0
    for idx in range(sample_size):
        for jdx in range(k):
            if dist[idx]== -1:
                change+=1
                dist[idx] = distance(x[idx], means[jdx])
                cluster_no[idx] = jdx

            if dist[idx]!= -1 and dist[idx] > distance(x[idx], means[jdx]):
                change+=1
                dist[idx] = distance(x[idx], means[jdx])
                cluster_no[idx] = jdx


    means = [0 for idx in range(k)]
    for idx in range(k):
        means[idx] = [0 for i in range(dim)]
        count_means=0
        for jdx in range(sample_size):
            if cluster_no[jdx] == idx:
                means[idx] = vector_sum(means[idx], x[jdx])
                count_means+=1
        if count_means != 0.0:
            for idxx in range(dim):
                means[idx][idxx]/count_means
    if change == 0:
        break
#Plot different clusters with different colors
print_cluster_no = 2
y = [x[idx] for idx in range(sample_size) if cluster_no[idx] == print_cluster_no]

for idx in range(len(y)):
    print(y[idx])

#Finding the maximum diameter for each of the clusters
diam = [0 for idx in range(k)]

yy = [[] for idx in range(k)]

for idx in range(sample_size):
    yy[cluster_no[idx]].append(x[idx])

print(len(yy[1]))
print(len(yy))

for idx in range(k):
    for jdx in range(len(yy[idx])):
       for kdx in range(len(yy[idx])):
            if kdx > jdx:
                diam[idx] = max(diam[idx], distance(yy[idx][jdx], yy[idx][kdx]))

print(diam)

#Now we can do some plotting. Below we print the first four clusters.
x_axis = [yy[0][idx][0] for idx in range(len(yy[0]))]
y_axis = [yy[0][idx][1] for idx in range(len(yy[0]))]

plt.plot(x_axis,y_axis,'ro')
plt.show()

x_axis = [yy[1][idx][0] for idx in range(len(yy[1]))]
y_axis = [yy[1][idx][1] for idx in range(len(yy[1]))]
plt.plot(x_axis,y_axis,'ro', color = 'green')
plt.show()

x_axis = [yy[2][idx][0] for idx in range(len(yy[2]))]
y_axis = [yy[2][idx][1] for idx in range(len(yy[2]))]
plt.plot(x_axis,y_axis,'ro', color='blue')
plt.show()

x_axis = [yy[3][idx][0] for idx in range(len(yy[3]))]
y_axis = [yy[3][idx][1] for idx in range(len(yy[3]))]
plt.plot(x_axis, y_axis, 'ro', color = 'orange')
plt.show()