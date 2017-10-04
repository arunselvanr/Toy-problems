import numpy as np

#######################################################Simple array operations
a = np.array([1,2,3,4])
print(type(a))

print(a.shape[0])

b = np.array([[1,2,3,4], [4,5,6,7], [78,9.0,5,6]])
print(b.shape[1])

print(b[0][1])
print(b[2][1])


a = np.zeros((2,3))
b = np.ones((3,4))
c = np.full((4,5),6.00)
d = np.eye(6)
e = np.random.random((3,4))

print('Hello there %f'%(d[1,2]))

#######################################################Array Indexing
print('This portion is array indexing related')
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)
b = a[0:2, 2:3]
print(b)
b[1,0] = b[1,0] * 3
print(b)

#######################################################
c = a[0:2,2]
print(c.shape)
d = a[:, 2]
print(d.shape)
e = a[:, 2:3]
print(e.shape)
print(d[1])
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(a)
a[[0,2,3], np.arange(3)] += 10
print(a)
####################################################################Lets try out a few matrix operations.
x = np.array([[1,2],[3,4]], dtype = np.float64)
y = np.array([[5,6],[7,8]], dtype = np.float64)

v = np.array([9,10], dtype = np.float64)
w = np.array([11, 12], dtype = np.float64)

print(x)
print(y)
print(x + y)
print (x - y)
print(x*y)
print(x/y)
print(np.dot(x,v).shape)
print(type(np.dot(v,x)))
print(np.dot(x,y))
print(np.dot(y,x))